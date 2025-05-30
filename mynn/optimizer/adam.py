from .optimizer import Optimizer
from ..layer.blocks import BasicBlock

import cupy as cp


class Adam(Optimizer):
    """
        Adam Optimizer with Weight Decay Regularization

        带权重衰减的 Adam 优化器，本质上是 Adam 与 L2 正则化的结合。

    """
    def __init__(self, model=None, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(model=model, lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # 时间步
        self.m = []  # 均值
        self.v = []  # 方差

        # 初始化均值和方差
        for layer in self.model.layers:
            if layer.init['optimizable']:
                if isinstance(layer, BasicBlock):
                    # 为 BasicBlock 的子层创建优化器状态
                    block_m, block_v = {}, {}
                    for sublayer in [layer.conv1, layer.bn1, layer.conv2, layer.bn2,
                                    layer.short_conv, layer.short_bn]:
                        if sublayer is not None and sublayer.init['optimizable']:
                            for key in sublayer.params.keys():
                                block_m[id(sublayer), key] = cp.zeros_like(sublayer.params[key])
                                block_v[id(sublayer), key] = cp.zeros_like(sublayer.params[key])
                    self.m.append(block_m)
                    self.v.append(block_v)
                    continue

                layer_m = {}
                layer_v = {}
                for key in layer.params.keys():
                    layer_m[key] = cp.zeros_like(layer.params[key])
                    layer_v[key] = cp.zeros_like(layer.params[key])
                self.m.append(layer_m)
                self.v.append(layer_v)
            else:  # 占位符
                self.m.append(None)
                self.v.append(None)

    def step(self):
        self.t += 1

        for i, layer in enumerate(self.model.layers):
            if layer.init['optimizable']:
                if isinstance(layer, BasicBlock):
                    block_m = self.m[i]
                    block_v = self.v[i]
                    for sublayer in [layer.conv1, layer.bn1, layer.conv2, layer.bn2,
                                    layer.short_conv, layer.short_bn]:
                        if sublayer is not None and sublayer.init['optimizable']:
                            for key in sublayer.params.keys():
                                # 权重衰减
                                sublayer.params[key] *= (1 - self.lr * sublayer.init['weight_decay'])

                                grad = sublayer.grads[key]
                                m_key = (id(sublayer), key)
                                block_m[m_key] = self.beta1 * block_m[m_key] + (1 - self.beta1) * grad
                                block_v[m_key] = self.beta2 * block_v[m_key] + (1 - self.beta2) * (grad ** 2)

                                m_hat = block_m[m_key] / (1 - self.beta1 ** self.t)
                                v_hat = block_v[m_key] / (1 - self.beta2 ** self.t)

                                sublayer.params[key] -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)
                    continue

                layer_m = self.m[i]
                layer_v = self.v[i]
                for key in layer.params.keys():
                    # 权重衰减
                    layer.params[key] *= (1 - self.lr * layer.init['weight_decay'])

                    # 更新均值和方差
                    grad = layer.grads[key]
                    layer_m[key] = self.beta1 * layer_m[key] + (1 - self.beta1) * grad
                    layer_v[key] = self.beta2 * layer_v[key] + (1 - self.beta2) * (grad ** 2)

                    # 偏差修正
                    # 偏差修正
                    m_hat = layer_m[key] / (1 - self.beta1 ** self.t)
                    v_hat = layer_v[key] / (1 - self.beta2 ** self.t)

                    # 更新参数
                    layer.params[key] -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)
