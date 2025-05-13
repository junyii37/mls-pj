from ..layer import Layer
import cupy as cp

class BN(Layer):
    """ Batch Normalization layer，用于卷积层或全连接层的通用批归一化层 """
    def __init__(self,
                 normalized_dims,
                 param_shape,
                 epsilon=1e-5,
                 optimizable=True,
                 weight_decay=0.0):
        """
        normalized_dims: 需计算统计量的维度元组，例如 (0,2,3)
        param_shape:     γ,β 的形状，例如 (1,C,1,1) 或 (D,)
        epsilon:         数值稳定项
        optimizable:     是否学习 γ, β
        weight_decay:    L2 权重衰减系数，兼容 Optimizer
        """
        super().__init__()
        self.init = {
            'normalized_dims': normalized_dims,
            'param_shape': param_shape,
            'epsilon': epsilon,
            'optimizable': optimizable,
            'weight_decay': weight_decay,
        }

        # 在 __init__ 里立即初始化可学习参数，保证 Optimizer 能读到
        self.params = {}
        if optimizable:
            self.params['gamma'] = cp.ones(param_shape)
            self.params['beta']  = cp.zeros(param_shape)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        dims    = self.init['normalized_dims']
        epsilon = self.init['epsilon']

        M   = cp.prod(cp.array([X.shape[d] for d in dims]))
        mu  = cp.mean(X, axis=dims, keepdims=True)
        var = cp.var (X, axis=dims, keepdims=True)

        X_centered = X - mu
        std_inv    = 1.0 / cp.sqrt(var + epsilon)
        X_norm     = X_centered * std_inv

        self.cache = (X_norm, std_inv, M)

        if self.init['optimizable']:
            out = self.params['gamma'] * X_norm + self.params['beta']
        else:
            out = X_norm
        return out

    def backward(self, grad):
        dims    = self.init['normalized_dims']
        X_norm, std_inv, M = self.cache

        if self.init['optimizable']:
            self.grads['gamma'] = cp.sum(grad * X_norm, axis=dims, keepdims=True)
            self.grads['beta']  = cp.sum(grad, axis=dims, keepdims=True)
            grad_norm = grad * self.params['gamma']
        else:
            grad_norm = grad

        dY_sum        = cp.sum(grad_norm, axis=dims, keepdims=True)
        dY_Xnorm_sum = cp.sum(grad_norm * X_norm, axis=dims, keepdims=True)
        dX = (grad_norm
              - dY_sum / M
              - X_norm * dY_Xnorm_sum / M) * std_inv
        return dX

    def clear_grad(self):
        self.grads = {}
        if self.init['optimizable']:
            self.grads['gamma'] = None
            self.grads['beta']  = None

    def set_params(self, params):
        for key, val in params.items():
            self.params[key] = cp.array(val)
