from .layer import Layer
from .initialization import He
import cupy as cp
from cupy.lib.stride_tricks import sliding_window_view


def zero_pad(inputs, padding=0):
    """
    输入： (batch_size, channel, h, w)
    输出： (batch_size, channel, h_padded, w_padded)
    """
    return cp.pad(inputs, (
        (0, 0),
        (0, 0),
        (padding, padding),
        (padding, padding)
    ), mode='constant', constant_values=0)


def dilate(inputs, stride=1):
    """
    输入： (batch_size, channel, h, w)
    输出： (batch_size, channel, h_dilated, w_dilated)
    """
    # 对 grads 做膨胀
    N, C, H, W = inputs.shape
    shape = (
        N,
        C,
        stride * (H - 1) + 1,
        stride * (W - 1) + 1
    )
    X_dilated = cp.zeros(shape)
    for i in range(H):
        for j in range(W):
            X_dilated[..., i * stride, j * stride] = inputs[..., i, j]
    return X_dilated


def sliding(inputs, kernel, stride=1):
    """
    输入： (batch_size, channel, h, w)
    输出： (batch_size, channel, h_out, w_out, kernel, kernel)
    """
    windows = sliding_window_view(x=inputs, window_shape=(kernel, kernel), axis=(2, 3))
    return windows[:, :, ::stride, ::stride, :, :]


class Conv(Layer):
    """ 卷积层 """
    def __init__(self, in_channel, out_channel, kernel=3, stride=1, padding=1, initialize_method=He, weight_decay=0, optimizable=True):
        super().__init__()

        self.init = {
            'in_channel': in_channel,
            'out_channel': out_channel,
            'kernel': kernel,
            'stride': stride,
            'padding': padding,
            'initialize_method': initialize_method,
            'weight_decay': weight_decay,
            'optimizable': optimizable
        }

        # 初始化参数
        self.params = {
            'W': initialize_method(size=(out_channel, in_channel, kernel, kernel)),
            'b': initialize_method(size=(out_channel,))
        }

        a=1

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        输入： (batch_size, in_channel, h_in, w_in)
        输出： (batch_size, out_channel, h_out, w_out)
        """
        # 对 X 进行零填充
        X = zero_pad(inputs=X, padding=self.init['padding'])
        self.cache = X  # 缓存填充后的 X

        # 生成滑动窗口
        X_windows = sliding(inputs=X, kernel=self.init['kernel'], stride=self.init['stride'])

        # Y = W * X + b
        Y = cp.einsum('nchwuv,ocuv->nohw', X_windows, self.params['W']) + self.params['b'].reshape(1, -1, 1, 1)
        return Y

    def backward(self, grads):
        """
        输入： (batch_size, out_channel, h_out, w_out)
        输出： (batch_size, in_channel, h_in, w_in)

        同时会计算并存储参数的梯度。
        """
        # b 的反向传播
        self.grads['b'] = cp.sum(grads, axis=(0, 2, 3), keepdims=False)

        ## 对 grads 做膨胀
        grads_dilated = dilate(inputs=grads, stride=self.init['stride'])

        # W 的反向传播
        X = self.cache
        h_out_dilated = grads_dilated.shape[2]
        X_windows = sliding(inputs=X, kernel=h_out_dilated, stride=1)
        self.grads['W'] = cp.einsum('ncuvhw,nohw->ocuv', X_windows, grads_dilated)


        # X 的反向传播
        # 对膨胀后的 grads 进行填充
        grads_dilated_padded = zero_pad(inputs=grads_dilated, padding=self.init['kernel'] - 1)

        ## 对 W 做 180 度旋转
        W_rot180 = cp.rot90(self.params['W'], axes=(2, 3), k=2)

        ## 反向传播
        grads_dilated_padded_windows = sliding(inputs=grads_dilated_padded, kernel=self.init['kernel'], stride=1)
        dX_padded = cp.einsum('nohwuv,ocuv->nchw', grads_dilated_padded_windows, W_rot180)

        padding = self.init['padding']
        if padding > 0:
            return dX_padded[:, :, padding:-padding, padding:-padding]
        else:
            return dX_padded

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

    def set_params(self, params):
        """ 将可优化参数转换为 cupy 数组并赋值 """
        for key in params.keys():
            self.params[key] = cp.array(params[key])
