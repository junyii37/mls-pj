from ..layer import Layer


class Flatten(Layer):
    """ Flatten layer，展平多维数据，置与卷积层输出之后，全连接输入之前 """
    def __init__(self, optimizable=False):
        super().__init__()
        self.init = {
            'optimizable': optimizable  # 默认为不可优化网络层
        }

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        输入： (N, in_channel, H_in, W_in)
        输出： (N, in_channel * H_in * W_in)
        """
        self.cache = X
        return X.reshape(X.shape[0], -1)

    def backward(self, grads_in):
        """
        输入： (N, in_channel * H_in * W_in)
        输出： (N, in_channel, H_in, W_in)
        """
        return grads_in.reshape(self.cache.shape)
