#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import moco.builder
import moco.loader
import torch

import torch_npu

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
# import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from moco.loader import (SAR_dataset, Compose, ColorJitter, Normalize, ToTensor, RandomResizedCrop,
                         RandomApply, GaussianBlur, RandomHorizontalFlip, TwoCropsTransform, RandomGrayscale,decompose_collated_batch)
from moco.box_generator import SCRLBoxGenerator



model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=128,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=400, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all NPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.03,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[120, 160],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=1,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    # default="tcp://224.66.41.62:23456",
    default='env://',
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="hccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--npu", default=None, type=int, help="npu id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N NPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

# moco specific configs:
parser.add_argument(
    "--moco-dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--moco-k",
    default=65536,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)

# options for moco v2
parser.add_argument("--mlp", action="store_true", help="use mlp head")
parser.add_argument(
    "--aug-plus", action="store_true", help="use moco v2 data augmentation"
)
parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.npu is not None:
        warnings.warn(
            "You have chosen a specific NPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # ngpus_per_node = torch.cuda.device_count()
    nnpus_per_node = torch_npu.npu.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = nnpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=nnpus_per_node, args=(nnpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.npu, nnpus_per_node, args)


def main_worker(npu, nnpus_per_node, args):
    args.npu = npu
    acc1_list = []
    acc5_list = []
    loss_list = []
    # suppress printing if not master
    if args.multiprocessing_distributed and args.npu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.npu is not None:
        print("Use NPU: {} for training".format(args.npu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * nnpus_per_node + npu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim,
        args.moco_k,
        args.moco_m,
        args.moco_t,
        args.mlp,
    )
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.npu is not None:
            # torch.cuda.set_device(args.gpu)
            torch_npu.npu.set_device(args.npu)
            # model.cuda(args.gpu)
            model.npu(args.npu)
            # When using a single NPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of NPUs we have
            args.batch_size = int(args.batch_size / nnpus_per_node)
            args.workers = int((args.workers + nnpus_per_node - 1) / nnpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.npu]
            )
        else:
            # model.cuda()
            model.npu()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available NPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.npu is not None:
        # torch.cuda.set_device(args.gpu)
        torch_npu.npu.set_device(args.npu)
        # model = model.cuda(args.gpu)
        model = model.npu(args.npu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = nn.CrossEntropyLoss().npu(args.npu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.npu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                # loc = "cuda:{}".format(args.gpu)
                loc = "npu:{}".format(args.npu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data)
    normalize = Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # if args.aug_plus:
    #     # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    #     augmentation = [
    #         RandomResizedCrop(448, scale=(0.2, 1.0)),
    #         # transforms.RandomCrop(384),
    #         RandomApply(
    #             [ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
    #         ),
    #         RandomGrayscale(p=0.2),
    #         RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
    #         RandomHorizontalFlip(),
    #         ToTensor(),
    #         normalize,
    #     ]
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            RandomResizedCrop(448, scale=(0.2, 1.0)),
            # transforms.RandomCrop(384),
            RandomApply(
                [ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            RandomGrayscale(p=0.2),
            RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        print("MoCov1 augmentation!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        augmentation = [
            # transforms.RandomCrop(448),
            RandomResizedCrop(448, scale=(0.2, 1.0)),  #  第一版为512
            # transforms.RandomGrayscale(p=0.2),
            ColorJitter(0.4, 0.4, 0, 0),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
        print("MoCov1 augmentation")

    # train_dataset = datasets.ImageFolder(
    #     traindir, moco.loader.TwoCropsTransform(transforms.Compose(augmentation))
    # )
    train_dataset = SAR_dataset(
        image_dir=traindir, transform=TwoCropsTransform(Compose(augmentation, with_trans_info=True))
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        avg_loss = train(train_loader, model, criterion, optimizer, epoch, args, acc1_list, acc5_list)
        loss_list.append(avg_loss)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % nnpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename="checkpoint_{:04d}.pth".format(epoch),
            )
    if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % nnpus_per_node == 0
        ):
        acc1_list = np.array([tensor.item() for tensor in acc1_list])
        acc5_list = np.array([tensor.item() for tensor in acc5_list])
        np.save('acc1_list.npy', acc1_list)
        np.save('acc5_list.npy', acc5_list)
        np.save('loss_list.npy', np.array(loss_list))

def train(train_loader, model, criterion, optimizer, epoch, args, acc1_list, acc5_list):
    num_patches_per_image = 10  # 在builder那里也要改
    box_generator = SCRLBoxGenerator(
        input_size=448,                                              # 注意改这里，数据增强改的话
        min_size=32,
        num_patches_per_image=num_patches_per_image,
        box_jittering=False,
        box_jittering_ratio=0,
        iou_threshold=0.5,
        grid_based_box_gen=True
    )

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    Similarity_Loss = AverageMeter("Similarity_Loss", ":.4e")
    Global_Loss = AverageMeter("Global_Loss", ":.4e")
    Low_Level_Loss = AverageMeter("Low_level_loss", ":.4e")
    Losses_Total = AverageMeter("Losses_Total", ":.4e")
    Global_Top1 = AverageMeter("global_Acc@1", ":6.2f")
    Global_Top5 = AverageMeter("global_Acc@5", ":6.2f")
    Low_Level_Top1 = AverageMeter("Low_Level_Acc@1", ":6.2f")
    Low_Level_Top5 = AverageMeter("Low_Level_Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, Similarity_Loss, Global_Loss, Low_Level_Loss, Losses_Total,
         Global_Top1, Global_Top5, Low_Level_Top1, Low_Level_Top5],
        prefix="Epoch: [{}]".format(epoch),
    )
    # switch to train mode
    model.train()
    end = time.time()
    for i, views in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.npu is not None:
            images, transf, _, _ = decompose_collated_batch(views)
            boxes = box_generator.generate(transf)
            images[0] = images[0].npu(args.npu, non_blocking=True)
            images[1] = images[1].npu(args.npu, non_blocking=True)
            box1 = boxes[0].npu(args.npu, non_blocking=True)
            box2 = boxes[1].npu(args.npu, non_blocking=True)
            # assert box1.shape[0] == images[0].shape[0] * 10
            batchsize = images[0].shape[0]
            box2[:, 0] -= batchsize

        # compute output
        output, target, output2, target2, similarity_loss = model(im_q=images[0], im_k=images[1], box1=box1, box2=box2)
        global_loss = criterion(output, target) * 0.8
        low_level_loss = criterion(output2, target2) * 0.2
        
        # if epoch < 100 :
        #     weight = 20
        # else:
        #     weight = (epoch+1) / 5
        weight = 5.0
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        global_acc1, global_acc5 = accuracy(output, target, topk=(1, 5))
        local_acc1, local_acc5 = accuracy(output2, target2, topk=(1, 5))
        acc1_list.append(global_acc1[0])
        acc5_list.append(global_acc5[0])

        similarity_loss = similarity_loss * weight
        Similarity_Loss.update(similarity_loss.item(), images[0].size(0))
        Global_Loss.update(global_loss.item(), images[0].size(0))
        Low_Level_Loss.update(low_level_loss.item(), images[0].size(0))

        losses_total = similarity_loss + global_loss + low_level_loss

        Losses_Total.update(losses_total.item(), images[0].size(0))

        Global_Top1.update(global_acc1[0], images[0].size(0))
        Global_Top5.update(global_acc5[0], images[0].size(0))
        Low_Level_Top1.update(local_acc1[0], images[0].size(0))
        Low_Level_Top5.update(local_acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        losses_total.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return Losses_Total.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
