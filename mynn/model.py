import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cupy as cp
import matplotlib.pyplot as plt
from matplotlib import rcParams

import mynn.layer as nn


def name_to_layer(name):
    """
    将字符串映射到 Layer 类。后续若添加新层，只需在此注册。
    """
    mapping = {
        'Linear': nn.Linear,
        'Conv': nn.Conv,
        'LeakyReLU': nn.LeakyReLU,
        'ReLU': nn.ReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'Softmax': nn.Softmax,
        'BN': nn.BN,
        'Dropout': nn.Dropout,
        'Flatten': nn.Flatten,
        'Pooling': nn.Pooling,
    }
    try:
        return mapping[name]
    except KeyError:
        raise ValueError(f"Unknown layer name: {name}")


class Model:
    """神经网络模型类，支持保存/加载和参数可视化。"""

    def __init__(self, layers = None):
        self.layers = layers

    def __call__(self, X, train=True):
        return self.forward(X, train=train)

    def forward(self, X, train=True):
        for layer in self.layers:
            # 根据 训练/测试 模式设定 Dropout 层的行为模式
            if isinstance(layer, nn.Dropout):
                layer.set_training(train)
            X = layer.forward(X)
        return X

    def backward(self, grads):
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def save_model(self, file_path):
        """
        保存模型结构与参数。
        """
        # 创建文件夹
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 构造 model_info 字典
        model_info = {
            'layers': [],
            'init': [],
            'params': []
        }

        # 保存模型
        for layer in self.layers:
            # 保存类名和初始化参数
            model_info['layers'].append(layer.__class__.__name__)
            model_info['init'].append(layer.init)
            # 将可优化参数转换为 numpy 数组并保存
            params = {k: v.get() for k, v in layer.params.items()}
            model_info['params'].append(params)

        # 写入文件
        with open(path, 'wb') as f:
            pickle.dump(model_info, f)

        print(f"### Best model saved to: {path}")

    def load_model(self, file_path):
        """
        从文件加载模型结构与参数，重置当前 layers 列表。
        Returns:
          self，方便链式调用
        """
        path = Path(file_path)

        with open(path, 'rb') as f:
            model_info = pickle.load(f)

        self.layers = []
        for name, init_args, params in zip(model_info['layers'],
                                           model_info['init'],
                                           model_info['params']):
            class_layer = name_to_layer(name)
            layer = class_layer(**init_args) if init_args else class_layer()
            if layer.init['optimizable']:
                layer.set_params(params)
            self.layers.append(layer)

        print(f"Model loaded from: {path}")
        # 返回模型
        return self

    def visualize_parameters(self, max_cols: int = 4, save_dir: Optional[str] = None) -> None:
        """
        遍历所有可训练参数，分别用直方图、热力图、卷积核图展示。
        Args:
          max_cols: 卷积核网格每行最大数量
          save_dir: 若指定，则将图像写入该目录，不仅仅 plt.show()
        """
        if save_dir:
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = None

        # 中文支持
        rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        rcParams['axes.unicode_minus'] = False

        for idx, lyr in enumerate(self.layers):
            if not hasattr(lyr, 'params') or lyr.params is None:
                continue

            for name, arr in lyr.params.items():
                data = arr.get()  # 转为 NumPy

                title = f"Layer {idx} ({lyr.__class__.__name__}) {name} {data.shape}"
                if data.ndim == 1:
                    self._plot_hist1d(data, title, out_dir, f"{idx}_{name}.png")
                elif data.ndim == 2:
                    self._plot_heatmap2d(data, title, out_dir, f"{idx}_{name}.png")
                elif data.ndim == 4:
                    self._plot_kernels4d(data, title, max_cols, out_dir, f"{idx}_{name}.png")
                else:
                    print(f"Skipping {data.ndim}D param: {name}")

    @staticmethod
    def _plot_hist1d(data, title, out_dir: Optional[Path], fname: str):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(data, bins=50, edgecolor='black')
        ax.set(title=title, xlabel="值", ylabel="频次")
        plt.tight_layout()
        if out_dir:
            fig.savefig(out_dir / fname)
        plt.show()

    @staticmethod
    def _plot_heatmap2d(data, title, out_dir: Optional[Path], fname: str):
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(data.T, aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.set(title=title, xlabel="in", ylabel="out")
        plt.tight_layout()
        if out_dir:
            fig.savefig(out_dir / fname)
        plt.show()

    @staticmethod
    def _plot_kernels4d(data, title, max_cols, out_dir: Optional[Path], fname: str):
        # 将 (C_out, C_in, H, W) 展平至 (C_out*C_in, H, W)
        kernels = data.reshape(-1, *data.shape[2:])
        num = kernels.shape[0]
        rows = min((num + max_cols - 1) // max_cols, 8)
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(title)
        # 直方图
        ax0 = fig.add_subplot(1, 2, 1)
        ax0.hist(kernels.flatten(), bins=50, edgecolor='black')
        ax0.set(title="整体直方图", xlabel="值", ylabel="频次")
        # 核可视化
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])
        sub_gs = gs[0,1].subgridspec(rows, max_cols, wspace=0.2, hspace=0.2)
        for i, ker in enumerate(kernels[:rows*max_cols]):
            ax = fig.add_subplot(sub_gs[i])
            ax.imshow(ker, cmap='coolwarm')
            ax.axis('off')
        plt.tight_layout()
        if out_dir:
            fig.savefig(out_dir / fname)
        plt.show()
