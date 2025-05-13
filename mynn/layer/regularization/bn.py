from ..layer import Layer

import cupy as cp


class BN(Layer):
    """ Batch Normalization layer，用于卷积层或全连接层的通用批归一化层 """
    def __init__(self, normalized_dims, epsilon=1e-5, optimizable=False):
        """
        需输入需要计算统计量的维度元组，例如：
        - 对于(N,C)输入使用(0,)
        - 对于(N,C,H,W)输入使用(0,2,3)
        """
        super().__init__()

        self.init = {
            'normalized_dims': normalized_dims,  # 需计算统计量的维度
            'epsilon': epsilon,  # 小量，提高数值计算稳定性
            'optimizable': optimizable  # 默认为不可优化网络层
        }

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        dims = self.init['normalized_dims']
        epsilon = self.init['epsilon']

        # 计算每个特征的样本数
        M = cp.prod(cp.array([X.shape[d] for d in dims]))

        # 计算均值和方差
        mu = cp.mean(X, axis=dims, keepdims=True)
        var = cp.var(X, axis=dims, keepdims=True)

        # 归一化
        X_centered = X - mu
        std_inv = 1.0 / cp.sqrt(var + epsilon)
        X_norm = X_centered * std_inv

        self.cache = X_norm, std_inv, M
        return X_norm


    def backward(self, grad):
        dims = self.init['normalized_dims']
        X_norm, std_inv, M = self.cache

        # 计算梯度
        dY_sum = cp.sum(grad, axis=dims, keepdims=True)
        dY_Xnorm_sum = cp.sum(grad * X_norm, axis=dims, keepdims=True)

        return (grad - dY_sum / M - X_norm * dY_Xnorm_sum / M) * std_inv