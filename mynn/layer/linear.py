from .layer import Layer
from .initialization import He
import cupy as cp


class Linear(Layer):
    """ 全连接线性层 """
    def __init__(self, in_channel, out_channel, initialize_method=He, weight_decay=0, optimizable=True):
        super().__init__()

        self.init = {
            'in_channel': in_channel,
            'out_channel': out_channel,
            'initialize_method': initialize_method,
            'weight_decay': weight_decay,
            'optimizable': optimizable
        }

        # 初始化参数
        self.params = {
            'W': initialize_method(size=(in_channel, out_channel)),
            'b': initialize_method(size=(out_channel, ))  # b 是一维的
        }

        a=1

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        输入： (batch_size, in_channel)
        输出： (batch_size, out_channel)
        """
        self.cache = X
        # 前向传播
        Y = cp.dot(X, self.params['W']) + self.params['b'].reshape(1, -1)
        return Y

    def backward(self, grads):
        """
        输入： (batch_size, out_channel)
        输出： (batch_size, in_channel)

        同时会计算并存储参数的梯度。
        """
        X = self.cache
        # 可更新参数梯度
        self.grads = {
            'W': cp.dot(X.T, grads),
            'b': cp.sum(grads, axis=0, keepdims=False)
        }

        # 返回输入梯度
        return cp.dot(grads, self.params['W'].T)

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

    def set_params(self, params):
        """ 将可优化参数转换为 cupy 数组并赋值 """
        for key in params.keys():
            self.params[key] = cp.array(params[key])
