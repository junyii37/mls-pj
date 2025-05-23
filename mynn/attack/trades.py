import cupy as cp
import numpy as np
from tqdm import tqdm


# def pgd_kl_attack(model, images, epsilon=8 / 255, num_steps=10, step_size=2 / 255, batch_size=64):
#     num_samples = images.shape[0]
#     adv_images = []
#
#     # 预计算原始输出的概率分布（需固定参数）
#     with cp.cuda.Device(images.device):
#         orig_outputs = []
#         for i in range(0, num_samples, batch_size):
#             x_batch = images[i:i + batch_size]
#             logits = model.forward(x_batch, train=False)
#             orig_outputs.append(logits.copy())
#         orig_outputs = cp.concatenate(orig_outputs, axis=0)
#
#     # 使用 tqdm 进度条显示每个批次的处理进度
#     for i in tqdm(range(0, num_samples, batch_size), desc="Processing batches", ncols=100):
#         batch_start = i
#         batch_end = min(i + batch_size, num_samples)
#         x_orig = images[batch_start:batch_end].copy()
#         x_adv = x_orig.copy()
#         p_orig = orig_outputs[batch_start:batch_end]
#
#         for _ in range(num_steps):
#             # 前向传播（启用训练模式）
#             logits_adv = model.forward(x_adv, train=True)
#
#             # 计算KL散度梯度（手动计算d(KL)/d(logits_adv)）
#             p_adv = cp.clip(logits_adv, 1e-8, 1.0)
#             grad_kl = (p_adv - p_orig) / p_adv.shape[0]  # 关键修正：手动计算梯度
#
#             # 反向传播（传入正确形状的梯度）
#             model.backward(grads=grad_kl)  # grad_kl形状应与logits_adv一致
#
#             # 获取输入梯度
#             grad = model.grad_input
#
#             # 应用扰动
#             perturbation = step_size * cp.sign(grad)
#             x_adv = x_adv + perturbation
#
#             # 投影到epsilon邻域
#             delta = cp.clip(x_adv - x_orig, -epsilon, epsilon)
#             x_adv = x_orig + delta
#             x_adv = cp.clip(x_adv, 0.0, 1.0)
#
#         adv_images.append(x_adv)
#
#     return cp.concatenate(adv_images, axis=0)

def pgd_kl_attack(model, images, epsilon=8/255, num_steps=10, step_size=2/255,
                  batch_size=64, verbose=False):
    """
    TRADES专用PGD攻击生成对抗样本
    :param verbose: 是否显示进度条（测试时启用，训练时关闭）
    """
    num_samples = images.shape[0]
    adv_images = []

    # 预计算原始输出的概率分布（禁用训练模式）
    with cp.cuda.Device(images.device):
        orig_outputs = []
        for i in range(0, num_samples, batch_size):
            x_batch = images[i:i + batch_size]
            logits = model.forward(x_batch, train=False)  # 注意此处禁用Dropout等训练模式
            orig_outputs.append(logits.copy())
        orig_outputs = cp.concatenate(orig_outputs, axis=0)

    # 控制进度条显示逻辑
    batch_iter = range(0, num_samples, batch_size)
    if verbose:
        batch_iter = tqdm(batch_iter, desc="Generating Adversarial Examples", ncols=100)

    # 主循环
    for i in batch_iter:
        batch_start = i
        batch_end = min(i + batch_size, num_samples)
        x_orig = images[batch_start:batch_end].copy()
        x_adv = x_orig.copy()
        p_orig = orig_outputs[batch_start:batch_end]

        # PGD迭代（不显示内部进度条）
        for _ in range(num_steps):
            logits_adv = model.forward(x_adv, train=True)
            p_adv = cp.clip(logits_adv, 1e-8, 1.0)
            grad_kl = (p_adv - p_orig) / p_adv.shape[0]

            model.backward(grads=grad_kl)
            grad = model.grad_input

            perturbation = step_size * cp.sign(grad)
            x_adv = x_adv + perturbation
            delta = cp.clip(x_adv - x_orig, -epsilon, epsilon)
            x_adv = x_orig + delta
            x_adv = cp.clip(x_adv, 0.0, 1.0)

        adv_images.append(x_adv)

    return cp.concatenate(adv_images, axis=0)