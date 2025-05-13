from ..layer import Layer

import cupy as cp


class Sigmoid(Layer):
    """ Sigmoid layer，用于处理二分类问题 """
    def __init__(self, optimizable=False):
        super().__init__()

        self.init = {
            'optimizable': optimizable  # 默认为不可优化网络层
        }

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.cache = X
        return 1 / (1 + cp.exp(-X))

    def backward(self, grads):
        sigmoid = 1 / (1 + cp.exp(-self.cache))  # 前向传递结果
        self.cache = None
        return sigmoid * (1 - sigmoid) * grads  # Sigmoid 的导数
