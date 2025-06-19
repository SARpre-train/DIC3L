import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Dict
from collections import OrderedDict
from torch import Tensor
# import torchvision.models as models
# from torch_npu.contrib.module import ROIAlign


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


# class CustomResNet(nn.Module):
#     def __init__(self, base_encoder, dim=128):
#         super(CustomResNet, self).__init__()
#         # 加载预训练的ResNet-18模型
#         resnet = base_encoder(pretrained=False)
#         # 分离出ResNet的不同部分
#         self.features = nn.Sequential(*list(resnet.children())[:-2])  # 去掉全局平均池化和全连接层
#         self.avgpool = resnet.avgpool
#         self.global_projector = TwoLayerLinearHead(input_size=2048, hidden_size=2048, output_size=dim)
#         # self.local_projector = TwoLayerLinearHead(input_size=2048, hidden_size=2048, output_size=dim)
#         # self.roi_align = ROIAlign(output_size=(1, 1), sampling_ratio=-1, spatial_scale=0.03125, aligned=False)
#
#     def forward(self, x):
#         # 通过卷积层提取特征
#         features = self.features(x)
#         # 全局平均池化
#         out = self.avgpool(features)
#         out = torch.flatten(out, 1)
#         # 全连接层
#         out1 = self.global_projector(out)
#         # roi_out = self.roi_align(features, boxes).squeeze()
#         # out2 = self.local_projector(roi_out)
#         return out1, features
class CustomResNet(nn.Module):
    def __init__(self, base_encoder, dim=128):
        super(CustomResNet, self).__init__()
        resnet = base_encoder(pretrained=True)
        return_layers = {'layer3': 'low_feature', 'layer4': 'out'}
        self.encoder = IntermediateLayerGetter(resnet, return_layers=return_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.low_feature_projector = TwoLayerLinearHead(input_size=1024, hidden_size=2048, output_size=dim)
        self.global_feature_projector = TwoLayerLinearHead(input_size=2048, hidden_size=2048, output_size=dim)

    def forward(self, x):
        output = self.encoder(x)
        low_feature = output['low_feature']
        out = output['out']

        low_feature = torch.flatten(self.avgpool(low_feature), 1)
        low_feature = self.low_feature_projector(low_feature)

        global_feature = torch.flatten(self.avgpool(out), 1)
        global_feature = self.global_feature_projector(global_feature)

        return global_feature, low_feature, out


class TwoLayerLinearHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerLinearHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            # nn.ReLU(inplace=True),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, bias=True)
            # suppress bias for the last layer as in the BYOL official code
        )

        # Apply custom initialization
        # self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming initialization for linear layers
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # Initialization for BatchNorm1d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class TwoLayerLinearHead_BN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerLinearHead_BN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            # nn.ReLU(inplace=True),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, bias=True)
            # suppress bias for the last layer as in the BYOL official code
        )

        # Apply custom initialization
        # self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming initialization for linear layers
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # Initialization for BatchNorm1d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)    
    