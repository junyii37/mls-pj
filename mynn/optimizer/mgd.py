from .optimizer import Optimizer
from ..layer.blocks import BasicBlock

import cupy as cp


class MomentGD(Optimizer):
    def __init__(self, model=None, lr=0.01, mu=0.9):
        super().__init__(model=model, lr=lr)
        self.mu = mu  # 动量(Momentum)系数
        self.velocities = []

        # 初始化 velocities
        for layer in self.model.layers:
            if layer.init['optimizable']:
                if isinstance(layer, BasicBlock):
                    block_vel = {}
                    for sublayer in [layer.conv1, layer.bn1, layer.conv2, layer.bn2,
                                    layer.short_conv, layer.short_bn]:
                        if sublayer is not None and sublayer.init['optimizable']:
                            for key in sublayer.params.keys():
                                block_vel[(id(sublayer), key)] = cp.zeros_like(sublayer.params[key])
                    self.velocities.append(block_vel)
                    continue

                layer_vel = {}
                for key in layer.params.keys():
                    layer_vel[key] = cp.zeros_like(layer.params[key])
                self.velocities.append(layer_vel)
            else:
                self.velocities.append(None)  # 不可优化网络层的占位符

    def step(self):
        for i, layer in enumerate(self.model.layers):
            if layer.init['optimizable']:
                if isinstance(layer, BasicBlock):
                    block_vel = self.velocities[i]
                    for sublayer in [layer.conv1, layer.bn1, layer.conv2, layer.bn2,
                                    layer.short_conv, layer.short_bn]:
                        if sublayer is not None and sublayer.init['optimizable']:
                            for key in sublayer.params.keys():
                                grad = sublayer.grads[key]
                                v_key = (id(sublayer), key)

                                # 权重衰减
                                sublayer.params[key] *= (1 - self.lr * sublayer.init['weight_decay'])

                                # 动量更新
                                block_vel[v_key] = self.mu * block_vel[v_key] + grad

                                # 参数更新
                                sublayer.params[key] -= self.lr * block_vel[v_key]
                    continue  # BasicBlock 已处理，跳过
                
                layer_vel = self.velocities[i]
                for key in layer.params.keys():
                    # 权重衰减
                    layer.params[key] *= (1 - self.lr * layer.init['weight_decay'])

                    # v_t = beta * v_t-1 + grad
                    grad = layer.grads[key]
                    layer_vel[key] = self.mu * layer_vel[key] + grad

                    # w_t+1 = w_t - lr * v_t
                    layer.params[key] -= self.lr * layer_vel[key]