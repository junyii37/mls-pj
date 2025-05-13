from ..layer import Layer

import cupy as cp


class LeakyReLU(Layer):
    """ Leaky Rectified Linear Unit layer，解决传统 ReLU 的神经元死亡问题 """
    def __init__(self, alpha=0.01, optimizable=False):
        super().__init__()

        self.init = {
            'alpha': alpha,  # 设置负区域的斜率
            'optimizable': optimizable  # 默认为不可优化网络层
        }

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.cache = X
        return cp.where(X > 0, X, self.init['alpha'] * X)

    def backward(self, grads):
        X = self.cache
        return (X > 0).astype(X.dtype) * grads + (X <= 0).astype(X.dtype) * self.init['alpha'] * grads  # LeakyReLU 的导数
