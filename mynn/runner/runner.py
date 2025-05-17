import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cupy as cp
import sys
from tqdm import tqdm

from mynn.data import dataloader
from ..attack import generate_adversarial_batch_bim, generate_adversarial_batch_fgsm

class RunnerM:
    """
    训练、评估、保存、加载模型。

    - 每次 train() 清空历史结果
    - 精确计算迭代次数
    - GPU 数值转为 Python float 存储
    - 使用 pathlib 管理路径
    - 保存训练曲线图
    """
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

        self.results = defaultdict(list)
        self.path: Path = None
        self.num_epochs: int = 0

    def train(self, train_set, dev_set, **kwargs):
        # 参数 & 路径准备
        batch_size = kwargs.get('batch_size', 128)
        num_epochs = kwargs.get('num_epochs', 50)
        scheduler = kwargs.get('scheduler', None)
        strategy = kwargs.get('strategy', None)
        shuffle = kwargs.get('shuffle', True)
        save_dir = Path(kwargs.get('save_dir', 'saved_models'))

        # 重置结果，支持多次调用
        self.results.clear()

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = save_dir / now
        self.path.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs

        best_score = 0.0
        total_iters = math.ceil(len(train_set[0]) / batch_size)

        # 训练循环
        for epoch in range(1, num_epochs + 1):
            desc = f"Epoch {epoch}/{num_epochs}"

            loader = dataloader(train_set, batch_size, shuffle)
            with tqdm(loader,
                      total=total_iters,
                      desc=desc,
                      file=sys.stdout,
                      unit='batch',
                      colour='red') as pbar:

                for X_batch, y_batch in pbar:
                    # 前向
                    logits = self.model(X_batch, train=True)
                    loss_val, acc_val = self.loss(logits, y_batch)

                    # 转 float 存储
                    loss_f = float(loss_val)
                    acc_f = float(acc_val)
                    self.results['train_loss_iter'].append(loss_f)
                    self.results['train_acc_iter'].append(acc_f)

                    # 反向 & 更新
                    self.loss.backward()
                    self.optimizer.step()

                    # 更新 postfix
                    pbar.set_postfix(loss=loss_f, accuracy=acc_f)

            # Epoch 后评估
            train_loss, train_acc = self.evaluate(train_set)
            dev_loss,   dev_acc   = self.evaluate(dev_set)

            print(f"train_loss: {train_loss:.5f}, train_acc: {train_acc:.5f}")
            print(f"dev_loss  : {dev_loss:.5f}, dev_acc  : {dev_acc:.5f}")

            self.results['train_loss_epoch'].append(train_loss)
            self.results['train_acc_epoch'].append(train_acc)
            self.results['dev_loss_epoch'].append(dev_loss)
            self.results['dev_acc_epoch'].append(dev_acc)
            self.results['epoch_iters'].append(len(self.results['train_loss_iter']))

            if scheduler:
                scheduler.step()

            # 保存最优
            if dev_acc > best_score:
                save_path = self.path / 'best_model.pickle'
                self.save_model(str(save_path))
                print(f"### Best validation accuracy updated: {best_score:.5f} -> {dev_acc:.5f}")
                best_score = dev_acc

            if strategy and strategy.step(dev_acc):
                break

        # 画图并保存
        self.plot()

    def train_with_attack(self, train_set, dev_set, **kwargs):
        batch_size = kwargs.get('batch_size', 128)
        num_epochs = kwargs.get('num_epochs', 50)
        scheduler = kwargs.get('scheduler', None)
        strategy = kwargs.get('strategy', None)
        epsilon = kwargs.get('epsilon', 1.0)
        shuffle = kwargs.get('shuffle', True)
        attack_strategy = kwargs.get('attack_strategy', None)
        save_dir = Path(kwargs.get('save_dir', 'saved_models'))

        if attack_strategy == 'bim':
            generate_adversarial_batch = generate_adversarial_batch_bim
        else:
            generate_adversarial_batch = generate_adversarial_batch_fgsm

        self.results.clear()
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = save_dir / now
        self.path.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs

        best_score = 0.0
        total_iters = math.ceil(len(train_set[0]) / batch_size)

        for epoch in range(1, num_epochs + 1):
            desc = f"[Adversarial Train] Epoch {epoch}/{num_epochs}"
            loader = dataloader(train_set, batch_size, shuffle)

            with tqdm(loader, total=total_iters, desc=desc, file=sys.stdout, unit='batch', colour='blue') as pbar:
                for X_batch, y_batch in pbar:
                    # 分出一半用于对抗样本
                    half = X_batch.shape[0] // 2
                    x_clean = X_batch[:half]
                    y_clean = y_batch[:half]
                    x_target = X_batch[half:]
                    y_target = y_batch[half:]

                    # 动态生成对抗样本
                    x_adv = generate_adversarial_batch(self.model, x_target, y_target, self.loss,epsilon)

                    # 合并 clean + adv
                    X_aug = cp.concatenate([x_clean, x_adv], axis=0)
                    y_aug = cp.concatenate([y_clean, y_target], axis=0)

                    # 前向 & 损失
                    logits = self.model(X_aug, train=True)
                    loss_val, acc_val = self.loss(logits, y_aug)

                    # 反向 & 更新
                    self.loss.backward()
                    self.optimizer.step()

                    # 存储日志
                    loss_f = float(loss_val)
                    acc_f = float(acc_val)
                    self.results['train_loss_iter'].append(loss_f)
                    self.results['train_acc_iter'].append(acc_f)
                    pbar.set_postfix(loss=loss_f, accuracy=acc_f)

            # 每 epoch 后评估
            train_loss, train_acc = self.evaluate(train_set)
            dev_loss, dev_acc = self.evaluate(dev_set)

            print(f"train_loss: {train_loss:.5f}, train_acc: {train_acc:.5f}")
            print(f"dev_loss  : {dev_loss:.5f}, dev_acc  : {dev_acc:.5f}")

            self.results['train_loss_epoch'].append(train_loss)
            self.results['train_acc_epoch'].append(train_acc)
            self.results['dev_loss_epoch'].append(dev_loss)
            self.results['dev_acc_epoch'].append(dev_acc)
            self.results['epoch_iters'].append(len(self.results['train_loss_iter']))

            if scheduler:
                scheduler.step()

            if dev_acc > best_score:
                save_path = self.path / 'best_model.pickle'
                self.save_model(str(save_path))
                print(f"### Best validation accuracy updated: {best_score:.5f} -> {dev_acc:.5f}")
                best_score = dev_acc

            if strategy and strategy.step(dev_acc):
                break

        self.plot()

    def evaluate(self, dataset, batch_size=256):
        loss_list, acc_list = [], []
        for X_batch, y_batch in dataloader(dataset, batch_size=batch_size, shuffle=False):
            Xb = cp.asarray(X_batch)
            yb = cp.asarray(y_batch)
            logits = self.model(Xb, train=False)
            loss_val, acc_val = self.loss(logits, yb)
            loss_list.append(float(loss_val))
            acc_list.append(float(acc_val))
        avg_loss = sum(loss_list) / len(loss_list) if loss_list else 0.0
        avg_acc  = sum(acc_list)  / len(acc_list)  if acc_list  else 0.0
        return avg_loss, avg_acc

    def save_model(self, save_path):
        self.model.save_model(save_path)

    def plot(self):
        """绘制训练过程可视化图表，包含四个子图：
        1. 迭代级训练损失    2. 迭代级训练精度
        3. Epoch级损失对比  4. Epoch级精度对比
        """
        from matplotlib import rcParams
        import matplotlib.pyplot as plt
        from pathlib import Path

        # 设置中文字体
        rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        rcParams['axes.unicode_minus'] = False

        data = self.results  # 获取训练结果数据
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 创建2x2子图布局
        plt.subplots_adjust(hspace=0.3, wspace=0.25)  # 调整子图间距

        # ================== 第一行：迭代级指标 ==================
        # 子图1：迭代级训练损失
        ax = axes[0, 0]
        if data['train_loss_iter']:
            ax.plot(
                data['train_loss_iter'],
                label='批次训练损失',
                alpha=0.6,
                linewidth=1,
                color='royalblue'
            )
            ax.set(
                xlabel='训练迭代次数',
                ylabel='批次损失',
                title='批次训练损失变化趋势',
                yscale='log' if max(data['train_loss_iter']) > 1e3 else 'linear'
            )
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        # 子图2：迭代级训练精度
        ax = axes[0, 1]
        if data.get('train_acc_iter'):
            ax.plot(
                data['train_acc_iter'],
                label='批次训练精度',
                alpha=0.6,
                linewidth=1,
                color='darkorange'
            )
            ax.set(
                xlabel='训练迭代次数',
                ylabel='批次精度',
                title='批次训练精度变化趋势',
                ylim=(0, 1) if max(data['train_acc_iter']) <= 1 else None
            )
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

            # ================== 第二行：Epoch级指标 ==================
            # 子图3：Epoch级损失对比
            ax = axes[1, 0]
            if data.get('epoch_iters'):
                epochs = data['epoch_iters']
                # 训练损失
                ax.plot(
                    epochs, data['train_loss_epoch'],
                    'o-', color='royalblue',
                    markersize=4, linewidth=2,
                    label='训练损失'
                )
                # 验证损失
                if data.get('dev_loss_epoch'):
                    ax.plot(
                        epochs, data['dev_loss_epoch'],
                        's--', color='crimson',
                        markersize=4, linewidth=2,
                        label='验证损失'
                    )
                ax.set(
                    xlabel='累计迭代次数',
                    ylabel='损失',
                    title='训练集&验证集 损失对比',
                    yscale='log' if max(data['train_loss_epoch']) > 1e3 else 'linear'
                )
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()

            # 子图4：Epoch级精度对比（优化纵坐标范围）
            ax = axes[1, 1]
            if data.get('epoch_iters'):
                epochs = data['epoch_iters']
                # 训练精度
                train_acc = data['train_acc_epoch']
                ax.plot(
                    epochs, train_acc,
                    'o-', color='darkorange',
                    markersize=4, linewidth=2,
                    label='训练精度'
                )
                # 验证精度
                dev_acc = data.get('dev_acc_epoch')
                if dev_acc:
                    ax.plot(
                        epochs, dev_acc,
                        's--', color='forestgreen',
                        markersize=4, linewidth=2,
                        label='验证精度'
                    )

                # ========== 关键优化：动态计算纵坐标范围 ==========
                all_acc = train_acc.copy()
                if dev_acc:
                    all_acc += dev_acc
                if all_acc:
                    min_acc = min(all_acc)
                    max_acc = max(all_acc)
                    padding = 0.05  # 上下留白 5%
                    # 动态调整下界，不低于 min(0, 数据最小值 - padding)
                    y_lower = max(0.0, min_acc - padding) if min_acc > 0 else 0.0
                    # 上界不超过 1.05（兼容百分比和 0~1 格式）
                    y_upper = min(1.05, max_acc + padding) if max_acc <= 1.1 else max_acc + padding
                    ax.set_ylim(y_lower, y_upper)

                ax.set(
                    xlabel='累计迭代次数',
                    ylabel='精度',
                    title='训练集&验证集 精度对比',
                )
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()

        # ================== 保存与展示 ==================
        plt.tight_layout()
        fig_path = self.path / 'training_metrics.png'
        fig.savefig(str(fig_path), bbox_inches='tight', dpi=150)
        plt.show()
