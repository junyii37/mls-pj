from ..layer import Layer

import cupy as cp


class Tanh(Layer):
    """ Tanh layer，不常用的激活函数 """
    def __init__(self, optimizable=False):
        super().__init__()

        self.init = {
            'optimizable': optimizable  # 默认为不可优化网络层
        }

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.cache = X
        return cp.tanh(X)

    def backward(self, grads):
        tanh = cp.tanh(self.cache)  # 前向传递结果
        self.cache = None
        return (1 - tanh ** 2) * grads  # Tanh 的导数
