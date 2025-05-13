from .optimizer import Optimizer

import cupy as cp


class MomentGD(Optimizer):
    def __init__(self, model=None, lr=0.01, mu=0.9):
        super().__init__(model=model, lr=lr)
        self.mu = mu  # 动量(Momentum)系数
        self.velocities = []

        # 初始化 velocities
        for layer in self.model.layers:
            if layer.init['optimizable']:
                layer_vel = {}
                for key in layer.params.keys():
                    layer_vel[key] = cp.zeros_like(layer.params[key])
                self.velocities.append(layer_vel)
            else:
                self.velocities.append(None)  # 不可优化网络层的占位符

    def step(self):
        for i, layer in enumerate(self.model.layers):
            if layer.init['optimizable']:
                layer_vel = self.velocities[i]
                for key in layer.params.keys():
                    # 权重衰减
                    layer.params[key] *= (1 - self.lr * layer.init['weight_decay'])

                    # v_t = beta * v_t-1 + grad
                    grad = layer.grads[key]
                    layer_vel[key] = self.mu * layer_vel[key] + grad

                    # w_t+1 = w_t - lr * v_t
                    layer.params[key] -= self.lr * layer_vel[key]