# Copyright (c) 2025, Shaanxi Yuanyi Intelligent Technology Co., Ltd.
# This file is part of a project licensed under the MIT License.
# It is developed based on the MoCo project by Meta Platforms, Inc.
# Original MoCo repository: https://github.com/facebookresearch/moco
#
# This project includes significant modifications tailored for SAR land-cover classification,
# including the design of domain-specific modules and the use of large-scale SAR datasets
# to improve performance and generalization on downstream SAR tasks.


from typing import NamedTuple, List, Tuple
from functools import wraps
import torch
import random
from torchvision.transforms import (ColorJitter, Normalize, ToTensor, RandomGrayscale)
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import ImageOps
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import os

class ImageWithTransInfo(NamedTuple):
    """to improve readability"""
    image: torch.Tensor  # image
    transf: List  # cropping coord. in the original image + flipped or not
    ratio: List  # resizing ratio w.r.t. the original image
    size: List  # size (width, height) of the original image


def free_pass_trans_info(func):
    """Wrapper to make the function bypass the second argument(transf)."""

    @wraps(func)
    def decorator(img, transf, ratio):
        return func(img), transf, ratio

    return decorator


def _with_trans_info(transform):
    """use with_trans_info function if possible, or wrap original __call__."""
    if hasattr(transform, 'with_trans_info'):
        transform = transform.with_trans_info
    else:
        transform = free_pass_trans_info(transform)
    return transform


def _get_size(size):
    if isinstance(size, int):
        oh, ow = size, size
    else:
        oh, ow = size
    return oh, ow


def _update_transf_and_ratio(transf_global, ratio_global,
                             transf_local=None, ratio_local=None):
    if transf_local:
        i_global, j_global, *_ = transf_global
        i_local, j_local, h_local, w_local = transf_local
        i = int(round(i_local / ratio_global[0] + i_global))
        j = int(round(j_local / ratio_global[1] + j_global))
        h = int(round(h_local / ratio_global[0]))
        w = int(round(w_local / ratio_global[1]))
        transf_global = [i, j, h, w]

    if ratio_local:
        ratio_global = [g * l for g, l in zip(ratio_global, ratio_local)]

    return transf_global, ratio_global


class Compose(object):
    def __init__(self, transforms, with_trans_info=False, seed=None):
        self.transforms = transforms
        self.with_trans_info = with_trans_info
        self.seed = seed

    @property
    def with_trans_info(self):
        return self._with_trans_info

    @with_trans_info.setter
    def with_trans_info(self, value):
        self._with_trans_info = value

    def __call__(self, *args, **kwargs):
        if self.with_trans_info:
            return self._call_with_trans_info(*args, **kwargs)
        return self._call_default(*args, **kwargs)

    def _call_default(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def _call_with_trans_info(self, img):
        w, h = img.size
        transf = [0, 0, h, w]
        ratio = [1., 1.]

        for t in self.transforms:
            t = _with_trans_info(t)
            try:
                if self.seed:
                    random.seed(self.seed)
                    torch.manual_seed(self.seed)
                img, transf, ratio = t(img, transf, ratio)
            except Exception as e:
                raise Exception(f'{e}: from {t.__self__}')

        return ImageWithTransInfo(img, transf, ratio, (h, w))


class CenterCrop(transforms.CenterCrop):
    def with_trans_info(self, img, transf, ratio):
        w, h = img.size
        oh, ow = _get_size(self.size)
        i = int(round((w - ow) * 0.5))
        j = int(round((h - oh) * 0.5))
        transf_local = [i, j, oh, ow]
        transf, ratio = _update_transf_and_ratio(
            transf, ratio, transf_local, None)
        return F.center_crop(img, self.size), transf, ratio


class Resize(transforms.Resize):
    def with_trans_info(self, img, transf, ratio):
        w, h = img.size  # PIL.Image
        resized_img = F.resize(img, self.size, self.interpolation)
        # get the size directly from resized image rather than using _get_size()
        # since only smaller edge of the image will be matched in this class.
        ow, oh = resized_img.size
        ratio_local = [oh / h, ow / w]
        transf, ratio = _update_transf_and_ratio(
            transf, ratio, None, ratio_local)
        return resized_img, transf, ratio


class RandomResizedCrop(transforms.RandomResizedCrop):
    def with_trans_info(self, img, transf, ratio):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        oh, ow = _get_size(self.size)
        transf_local = [i, j, h, w]
        ratio_local = [oh / h, ow / w]
        transf, ratio = _update_transf_and_ratio(
            transf, ratio, transf_local, ratio_local)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        return img, transf, ratio


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def with_trans_info(self, img, transf, ratio):
        if torch.rand(1) < self.p:
            transf.append(True)
            return F.hflip(img), transf, ratio
        transf.append(False)
        return img, transf, ratio


class RandomOrder(transforms.RandomOrder):
    def with_trans_info(self, img, transf, ratio):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            t = _with_trans_info(self.transforms[i])
            img, transf, ratio = t(img, transf, ratio)
        return img, transf, ratio


class RandomApply(transforms.RandomApply):
    def with_trans_info(self, img, transf, ratio):
        if self.p < random.random():
            return img, transf, ratio
        for t in self.transforms:
            t = _with_trans_info(t)
            img, transf, ratio = t(img, transf, ratio)
        return img, transf, ratio


class Solarize(object):
    def __init__(self, threshold):
        assert 0 < threshold < 1
        self.threshold = round(threshold * 256)

    def __call__(self, img):
        return ImageOps.solarize(img, self.threshold)

    def __repr__(self):
        attrs = f"(min_scale={self.threshold}"
        return self.__class__.__name__ + attrs


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class SAR_dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img_name) for img_name in os.listdir(image_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)  # 打开单通道图像
        img = Image.merge('RGB', [img, img, img])  # 复制为三通道图像
        images = self.transform(img)
        return images


def decompose_collated_batch(collated_batch):
    batch_views = []
    batch_transf = []
    batch_ratio = []
    batch_size = []
    if isinstance(collated_batch, ImageWithTransInfo):
        collated_batch = [collated_batch]
    for x in collated_batch:
        image, transf, ratio, size = x.image, x.transf, x.ratio, x.size
        batch_views.append(image)
        transf = torch.cat(transf).reshape(len(transf), image.size(0))
        transf = torch.transpose(transf, 1, 0)
        batch_transf.append(transf)
        batch_ratio.append(ratio)
        batch_size.append(size)
    return batch_views, batch_transf, batch_ratio, batch_size


# augmentation = [
#     RandomResizedCrop(512, scale=(0.6, 1.0)),
#     RandomApply(
#         [ColorJitter(0.4, 0.4, 0, 0)], p=0.8  # not strengthened
#     ),
#     # transforms.RandomGrayscale(p=0.2),
#     RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
#     RandomHorizontalFlip(),
#     ToTensor(),
#     Normalize(
#         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]
# train_dir = r"D:\BYOL-ALL-FILES\CROP_GRAY_PNG_095"
# train_dataset = SAR_dataset(
#     image_dir=train_dir, transform=TwoCropsTransform(Compose(augmentation, with_trans_info=True))
# )
# train_loader = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=1,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=True,
#     drop_last=True,
# )
#
# for step, views in enumerate(train_loader):
#     images, transf, _, _ = decompose_collated_batch(views)
#     image1 = images[0]
#     image2 = images[1]
#     transf1 = transf[0]
#     transf2 = transf[1]
#     pass


# ------------------------------------------------------------------------------------------
# 调试用

#
# def save_images(img, transf, transformed_img):
#     # 保存带红框的原图
#     img_with_box = img.copy()
#     draw = ImageDraw.Draw(img_with_box)
#     i, j, h, w, info = transf
#     draw.rectangle([j, i, j+w, i+h], outline="red")
#     img_with_box.save('original_with_box1.png')
#
#     # 保存变换后的图像
#     # transformed_img_pil = Image.fromarray(transformed_img)
#     transformed_img.save('transformed_image1.png')
#
#
# for step, views in enumerate(train_loader):
#     images, transf, _, _ = decompose_collated_batch(views)
#     image1 = Image.fromarray(np.transpose( (images[0].squeeze().numpy() * 255).astype(np.uint8), (1, 2, 0)) )
#     image2 = Image.fromarray(np.transpose( (images[1].squeeze().numpy() * 255).astype(np.uint8), (1, 2, 0)) )
#
#     transf1 = transf[0].squeeze().numpy()
#     transf2 = transf[1].squeeze().numpy()
#     print("1:{}".format(transf1))
#     print("2:{}".format(transf2))
#     image1.save("image1.png")
#     image2.save("image2.png")
#     print(step)
#     break

