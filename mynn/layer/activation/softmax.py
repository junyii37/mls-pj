from ..layer import Layer

import cupy as cp


class Softmax(Layer):
    """ Softmax layer，用于处理多分类问题。与交叉熵结合后，作为 MultiCrossEntropyLoss 层位于 mynn/layer/crossentropy.py 文件中 """
    def __init__(self, optimizable=False):
        super().__init__()

        self.init = {
            'optimizable': optimizable  # 默认为不可优化网络层
        }

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.cache = X
        exp_values = cp.exp(X - cp.max(X, axis=1, keepdims=True))  # 防止溢出
        return exp_values / cp.sum(exp_values, axis=1, keepdims=True)

    def backward(self, grads):
        # Softmax 的反向传播计算公式
        softmax = self.cache  # 先前计算的输出
        batch_size = softmax.shape[0]
        dX = cp.zeros_like(softmax)

        for i in range(batch_size):
            s = softmax[i, :]
            jacobian_matrix = cp.diagflat(s) - cp.outer(s, s)  # 雅可比矩阵
            dX[i, :] = cp.dot(jacobian_matrix, grads[i, :])  # 计算梯度

        return dX
