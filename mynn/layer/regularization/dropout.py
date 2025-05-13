from ..layer import Layer

import cupy as cp


class Dropout(Layer):
    """ Dropout layer，放置于全连接之前，在训练时随机禁用部分神经元，从而防止过拟合 """
    def __init__(self, rate, train=True, optimizable=False):
        super().__init__()

        self.init = {
            'rate': rate,  # 丢失率
            'train': train,  # 判断是否在训练
            'optimizable': optimizable  # 默认为不可优化网络层
        }

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        train = self.init['train']
        if train:
            # 生成掩码 mask
            mask = (cp.random.rand(*X.shape) > self.init['rate']).astype(cp.float32)
            self.cache = mask
            return X * mask / (1.0 - self.init['rate'])  # 保持期望输出不变
        else:
            # 在测试阶段，直接返回输入
            return X

    def backward(self, grad):
        mask = self.cache
        return grad * mask / (1.0 - self.init['rate'])

    def set_training(self, train):
        self.init['train'] = train
