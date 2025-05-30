import numpy as np
import cupy as cp

def generate_adversarial_batch_pgd(model, x_batch, y_batch, loss_fn, epsilon=2 / 255, num_steps=5, step_size=0.5 / 255):
    """
    基于 PGD 攻击方法生成对抗样本。
    """
    # 初始化随机扰动（与BIM的区别）
    x_adv = x_batch + cp.random.uniform(-epsilon, epsilon, x_batch.shape)
    x_adv = cp.clip(x_adv, 0, 1)

    for _ in range(num_steps):
        logits = model.forward(x_adv)
        loss, _ = loss_fn.forward(logits, y_batch)

        loss_fn.backward()
        grad = model.grad_input

        x_adv = x_adv + step_size * cp.sign(grad)

        # 保持扰动在 epsilon 内
        x_adv = cp.clip(x_adv, x_batch - epsilon, x_batch + epsilon)

        # 保证像素值在合法范围
        x_adv = cp.clip(x_adv, 0, 1)

    return x_adv

def pgd_attack(model, images, labels, loss_fn, epsilon=2 / 255, num_steps=5, step_size=0.5 / 255, batch_size=64):
    """
    对整个数据集进行 PGD 攻击（分批次）
    """
    num_samples = images.shape[0]
    adv_images = []

    images = cp.asarray(images)
    labels = cp.asarray(labels)

    for i in range(0, num_samples, batch_size):
        x_batch = images[i:i + batch_size]
        y_batch = labels[i:i + batch_size]

        x_adv = generate_adversarial_batch_pgd(model, x_batch, y_batch, loss_fn, epsilon, num_steps, step_size)
        adv_images.append(x_adv)

    adv_images = cp.concatenate(adv_images, axis=0)
    return adv_images

