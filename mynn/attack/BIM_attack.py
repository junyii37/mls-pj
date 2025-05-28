import numpy as np
import cupy as cp

def generate_adversarial_batch_bim(model, x_batch, y_batch, loss_fn, epsilon=2 / 255, num_steps=5, step_size=0.5 / 255):
    """
    基于 BIM 攻击方法生成对抗样本。

    参数:
        model: 模型对象
        x_batch: 输入图像，形状 (B, C, H, W)，cupy 数组
        y_batch: 标签，cupy 数组
        loss_fn: 损失函数对象，必须有 forward 和 backward 方法
        epsilon: 最大总扰动
        alpha: 每步扰动大小
        num_iter: 迭代次数

    返回:
        对抗样本（cupy 数组）
    """
    x_adv = x_batch.copy()

    for _ in range(num_steps):
        logits = model.forward(x_adv)
        loss, _ = loss_fn.forward(logits, y_batch)

        loss_fn.backward()
        grad = model.grad_input

        x_adv = x_adv + step_size * cp.sign(grad)

        x_adv = cp.clip(x_adv, x_batch - epsilon, x_batch + epsilon)

        x_adv = cp.clip(x_adv, 0, 1)

    return x_adv


def bim_attack(model, images, labels, loss_fn, epsilon=2 / 255, num_steps=5, step_size=0.5 / 255, batch_size=64):
    """
    对整个数据集进行 BIM 攻击（分批次）

    参数:
        model: 模型对象
        images: 原始图像数据，形状 (N, C, H, W)，范围 [0, 1]
        labels: 标签数据，形状 (N, ...)
        loss_fn: 损失函数对象
        epsilon: 最大扰动
        alpha: 每步更新幅度
        num_iter: 迭代次数
        batch_size: 批处理大小

    返回:
        对抗样本，cupy 数组
    """
    num_samples = images.shape[0]
    adv_images = []

    images = cp.asarray(images)
    labels = cp.asarray(labels)

    for i in range(0, num_samples, batch_size):
        x_batch = images[i:i + batch_size]
        y_batch = labels[i:i + batch_size]

        x_adv = generate_adversarial_batch_bim(model, x_batch, y_batch, loss_fn, epsilon, num_steps, step_size)
        adv_images.append(x_adv)

    adv_images = cp.concatenate(adv_images, axis=0)
    return adv_images