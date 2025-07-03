# Copyright (c) 2025, Shaanxi Yuanyi Intelligent Technology Co., Ltd.
# This file is part of a project licensed under the MIT License.
# It is developed based on the MoCo project by Meta Platforms, Inc.
# Original MoCo repository: https://github.com/facebookresearch/moco
#
# This project includes significant modifications tailored for SAR land-cover classification,
# including the design of domain-specific modules and the use of large-scale SAR datasets
# to improve performance and generalization on downstream SAR tasks.


import torch
import torch.nn as nn
from dic3l.net import CustomResNet, TwoLayerLinearHead, TwoLayerLinearHead_BN
import torch.nn.functional as F
from torch_npu.contrib.module import ROIAlign


def _regression_loss(preds, targets):
    bz = preds.size(0)
    preds_norm = F.normalize(preds, dim=1)
    targets_norm = F.normalize(targets, dim=1)
    loss = 2 - 2 * (preds_norm * targets_norm).sum() / bz
    return loss

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: dic3l momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # # create the encoders
        # # num_classes is the output fc dimension
        # self.encoder_q = base_encoder(num_classes=dim)
        # self.encoder_k = base_encoder(num_classes=dim)
        # self.encoder_q = base_encoder(pretrained=True)
        # self.encoder_k = base_encoder(pretrained=True)
        # in_features1 = self.encoder_q.fc.in_features
        # self.encoder_q.fc = torch.nn.Linear(in_features1, dim)
        # in_features2 = self.encoder_k.fc.in_features
        # self.encoder_k.fc = torch.nn.Linear(in_features2, dim)
        # print("pretrained=True")
        #
        # if mlp:  # hack: brute-force replacement
        #     dim_mlp = self.encoder_q.fc.weight.shape[1]
        #     self.encoder_q.fc = nn.Sequential(
        #         nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
        #     )
        #     self.encoder_k.fc = nn.Sequential(
        #         nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
        #     )
        if mlp:
            self.encoder_q = CustomResNet(base_encoder=base_encoder, dim=dim)
            self.encoder_k = CustomResNet(base_encoder=base_encoder, dim=dim)
        else:
            ValueError("only MoCoV2 in supported")

        self.local_projector_q = TwoLayerLinearHead_BN(input_size=2048, hidden_size=2048, output_size=dim)
        self.local_projector_k = TwoLayerLinearHead_BN(input_size=2048, hidden_size=2048, output_size=dim)


        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(
                self.local_projector_q.parameters(), self.local_projector_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.predictor = TwoLayerLinearHead_BN(input_size=dim, hidden_size=2048, output_size=dim)

        self.roi_align = ROIAlign(output_size=(1, 1), sampling_ratio=0, spatial_scale=0.03125, aligned=True)

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        for param_q, param_k in zip(
            self.local_projector_q.parameters(), self.local_projector_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, low_keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        low_keys = concat_all_gather(low_keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr+batch_size] = keys.T
        self.queue2[:, ptr:ptr+batch_size] = low_keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_npus = batch_size_all // batch_size_this

        # random shuffle index
        # idx_shuffle = torch.randperm(batch_size_all).cuda()
        idx_shuffle = torch.randperm(batch_size_all).npu()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        npu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_npus, -1)[npu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, box1, box2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q, q_low, features_q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        q_low = nn.functional.normalize(q_low, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k, k_low, features_k = self.encoder_k(im_k)  # keys: NxC

            k = nn.functional.normalize(k, dim=1)
            k_low = nn.functional.normalize(k_low, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k_low = self._batch_unshuffle_ddp(k_low, idx_unshuffle)
            features_k = self._batch_unshuffle_ddp(features_k, idx_unshuffle)

        q_local = self.roi_align(features_q, box1).squeeze()
        q_local = self.local_projector_q(q_local)

        with torch.no_grad():
            k_local = self.roi_align(features_k, box2).squeeze()
            k_local = self.local_projector_k(k_local)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        labels = torch.zeros(logits.shape[0], dtype=torch.long).npu()

        # 222222222222222222222222222222222222222222222
        l_pos2 = torch.einsum("nc,nc->n", [q_low, k_low]).unsqueeze(-1)
        # negative logits: NxK
        l_neg2 = torch.einsum("nc,ck->nk", [q_low, self.queue2.clone().detach()])

        # logits: Nx(1+K)
        logits2 = torch.cat([l_pos2, l_neg2], dim=1)

        # apply temperature
        logits2 /= self.T

        # labels: positive key indicators
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        labels2 = torch.zeros(logits2.shape[0], dtype=torch.long).npu()


        # dequeue and enqueue
        self._dequeue_and_enqueue(k, k_low)
        q_local_predictor = self.predictor(q_local)

        loss = _regression_loss(q_local_predictor, k_local)

        return logits, labels, logits2, labels2, loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
