import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# def mnist_augment(train=True):
#
#     # 自定义高斯噪声变换
#     class AddGaussianNoise(object):
#         def __init__(self, mean=0., std=0.05):
#             self.std = std
#             self.mean = mean
#
#         def __call__(self, tensor):
#             return tensor + torch.randn(tensor.size()) * self.std + self.mean
#
#         def __repr__(self):
#             return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
#
#
#     # 数据增强流程
#     mnist_train_transform = transforms.Compose([
#         # 将Tensor转换为PIL Image（MNIST.data的形状为[N, 28, 28]，值域0-255）
#         transforms.Lambda(lambda x: Image.fromarray(x.numpy(), mode='L')),
#
#         # 几何变换
#         transforms.RandomAffine(
#             degrees=15,  # 随机旋转角度范围
#             translate=(0.1, 0.1),  # 水平和垂直最大平移比例（10%）
#             scale=(0.9, 1.1),  # 随机缩放比例
#             interpolation=transforms.InterpolationMode.BILINEAR
#         ),
#
#         # 颜色/亮度调整
#         transforms.ColorJitter(brightness=0.1, contrast=0.1),
#
#         # 转换为Tensor并归一化到[0,1]
#         transforms.ToTensor(),
#
#         # 添加高斯噪声（在归一化前，输入为[0,1]）
#         AddGaussianNoise(std=0.05),
#
#         # MNIST标准化（均值0.1307，标准差0.3081）
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
#
#     # 测试集只需标准化
#     mnist_test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
#
#     if train:
#         return mnist_train_transform
#     else:
#         return mnist_test_transform

def mnist_augment(train=True):
    # 自定义高斯噪声变换（带数值裁剪）
    class AddGaussianNoise(object):
        def __init__(self, mean=0., std=0.03):
            self.std = std
            self.mean = mean

        def __call__(self, tensor):
            noisy_tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
            return torch.clamp(noisy_tensor, 0.0, 1.0)  # 确保数值在合理范围

        def __repr__(self):
            return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

    # 数据增强流程
    mnist_train_transform = transforms.Compose([
        # 将Tensor转换为PIL Image（保持与原始实现兼容）
        transforms.Lambda(lambda x: Image.fromarray(x.numpy(), mode='L')),

        # 几何变换（使用最近邻插值保持边缘锐利）
        transforms.RandomAffine(
            degrees=20,                  # 增大旋转角度范围
            translate=(0.15, 0.15),      # 增大平移比例
            scale=(0.8, 1.2),            # 扩展缩放范围
            shear=10,                    # 新增剪切变换
            interpolation=transforms.InterpolationMode.NEAREST
        ),

        # 随机透视变换（增强空间鲁棒性）
        transforms.RandomPerspective(
            distortion_scale=0.3,
            p=0.5,
            interpolation=transforms.InterpolationMode.NEAREST
        ),

        # 概率性颜色变换（仅亮度调整）
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.3, contrast=0.3)],
            p=0.5
        ),

        # 转换为Tensor
        transforms.ToTensor(),

        # 随机遮挡（增强抗遮挡能力）
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value='random'  # 使用0-1范围内的随机值
        ),

        # 高斯噪声（优化参数）
        AddGaussianNoise(std=0.03),

        # MNIST标准化
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 测试集保持基础变换
    mnist_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return mnist_train_transform if train else mnist_test_transform


def basic_mnist_augment(train=True):
    # 数据增强流程
    mnist_train_transform = transforms.Compose([
        # 将Tensor转换为PIL Image（MNIST.data的形状为[N, 28, 28]，值域0-255）
        transforms.Lambda(lambda x: Image.fromarray(x.numpy(), mode='L')),

        # 转换为Tensor并归一化到[0,1]
        transforms.ToTensor(),

        # MNIST标准化（均值0.1307，标准差0.3081）
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 测试集只需标准化
    mnist_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if train:
        return mnist_train_transform
    else:
        return mnist_test_transform


def merge_datasets(original_set, augment_set):
    """合并原始数据集与增强数据集"""
    # 解包数据
    orig_images, orig_labels = original_set
    aug_images, aug_labels = augment_set

    # 沿样本维度拼接
    merged_images = np.concatenate([orig_images, aug_images], axis=0)
    merged_labels = np.concatenate([orig_labels, aug_labels], axis=0)

    # 打乱顺序（重要！）
    shuffle_idx = np.random.permutation(len(merged_images))
    return merged_images[shuffle_idx], merged_labels[shuffle_idx]