from ..layer import Layer
from cupy.lib.stride_tricks import sliding_window_view as sliding


class Pooling(Layer):
    """ Pooling layer，池化层，降低分辨率，减少线性层的参数量，从而减轻了模型训练负担 """
    def __init__(self, kernel=2, optimizable=False):
        super().__init__()

        self.init = {
            'kernel': kernel,
            'optimizable': optimizable  # 默认为不可优化网络层
        }

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        X: (batch_size, in_channel, H_in, W_in)
        """
        K = self.init['kernel']
        H_in = X.shape[-2]
        if H_in % K != 0:
            raise ValueError("H_in 必须能被 kernel 整除。")

        # 生成滑动窗口
        windows = sliding(X, (K, K), axis=(2, 3))
        windows = windows[:, :, ::K, ::K, :, :]  # (N, in_channel/out_channel, H_out, W_out, kernel, kernel)

        X_pooling = windows.max(axis=(-2, -1), keepdims=True)
        mask = (windows == X_pooling)  # broadcasting

        self.cache = X.shape, mask
        return X_pooling.squeeze(-1).squeeze(-1)


    def backward(self, grads_in):
        """
        grads_in: (N, out_channel, H_out, W_out)
        """
        shape, mask = self.cache

        grads_in_expanded = grads_in[..., None, None] * mask
        grads_in_expanded = grads_in_expanded.transpose(0, 1, 2, 4, 3, 5)
        grads_out = grads_in_expanded.reshape(shape)

        self.cache = None
        return grads_out
