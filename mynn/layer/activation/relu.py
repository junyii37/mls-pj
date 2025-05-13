from ..layer import Layer

import cupy as cp


class ReLU(Layer):
    """ Rectified Linear Unit layer，常用的激活函数 """
    def __init__(self, optimizable=False):
        super().__init__()

        self.init = {
            'optimizable': optimizable # 默认为不可优化网络层
        }

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.cache = X
        return cp.maximum(0, X)

    def backward(self, grads):
        X = self.cache
        self.cache = None
        return (X > 0).astype(X.dtype) * grads
