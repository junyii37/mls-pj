import cupy as cp
import numpy as np


def generate_adversarial_batch_fgsm(model, x_batch, y_batch, loss_fn, epsilon=1 / 255):
    """
    给定 batch 数据，基于 FGSM 方法实时生成对抗样本。

    参数:
        model: 模型对象
        x_batch: 输入图像，形状 (B, C, H, W)，cupy 数组
        y_batch: 标签，cupy 数组
        epsilon: FGSM 扰动强度

    返回:
        对抗样本（cupy 数组）
    """
    # 前向
    logits = model.forward(x_batch)
    loss, _ = loss_fn.forward(logits, y_batch)

    # 反向求梯度
    loss_fn.backward()
    grad = model.grad_input

    # FGSM 扰动（像素值假设在 [0, 255]）
    x_adv = x_batch + epsilon * cp.sign(grad)
    x_adv = cp.clip(x_adv, 0, 1)

    return x_adv


def fgsm_attack(model, images, labels,loss_fn, epsilon=1/255, batch_size=64):
    """
    对整个测试集进行 FGSM 攻击（分批次）
    - images: (N, 3, 32, 32)
    - labels: (N, 10)
    """
    num_samples = images.shape[0]
    adv_images = []

    # 确保输入数据是cupy数组
    images = cp.asarray(images)
    labels = cp.asarray(labels)

    for i in range(0, num_samples, batch_size):
        # 分批次处理
        x_batch = images[i:i + batch_size]
        y_batch = labels[i:i + batch_size]

        x_batch = cp.asarray(x_batch)

        # 前向传播
        logits = model.forward(x_batch)  # 确保模型能够处理cupy数组输入
        loss, _ = loss_fn.forward(logits, y_batch)

        # 反向传播
        loss_fn.backward()
        grad = model.grad_input

        # FGSM 扰动
        x_adv = x_batch + epsilon * cp.sign(grad)  # 计算扰动
        x_adv = cp.clip(x_adv, 0, 1)

        adv_images.append(x_adv)

    # 拼接所有批次的对抗样本
    adv_images = cp.concatenate(adv_images, axis=0)
    return adv_images
