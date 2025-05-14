from ..layer import Layer
import cupy as cp

class BN(Layer):
    """批归一化层"""
    def __init__(
        self,
        normalized_dims,
        param_shape,
        epsilon=1e-5,
        optimizable=True,
        weight_decay=0.0,
        momentum=0.9,
        train=True
    ):
        super().__init__()
        # 初始化参数
        self.init = {
            'normalized_dims': normalized_dims,
            'param_shape':     param_shape,
            'epsilon':         epsilon,
            'optimizable':     optimizable,
            'weight_decay':    weight_decay,
            'momentum':        momentum,
            'train':           train,
        }
        # 可学习的缩放和平移参数
        self.params = {}
        if optimizable:
            self.params['gamma'] = cp.ones(param_shape)
            self.params['beta']  = cp.zeros(param_shape)
        # 训练过程中累积的均值和方差，用于测试时归一化
        self.running_mean = cp.zeros(param_shape, dtype=cp.float32)
        self.running_var  = cp.ones (param_shape, dtype=cp.float32)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        如果处于训练模式：
          - 计算当前 batch 的均值 mu 和方差 var
          - 更新 running_mean/ running_var
        否则：
          - 测试时使用 running 统计量
        然后：
          - 对输入做标准化
          - 按照 gamma 和 beta 生成输出
        """
        dims  = self.init['normalized_dims']
        eps   = self.init['epsilon']
        mom   = self.init['momentum']
        train = self.init['train']

        if train:
            # 计算 batch 统计量
            mu  = cp.mean(X, axis=dims, keepdims=True)
            var = cp.var (X, axis=dims, keepdims=True)
            # 更新移动平均
            self.running_mean = mom * self.running_mean + (1 - mom) * mu
            self.running_var  = mom * self.running_var  + (1 - mom) * var
        else:
            # 直接使用累积值
            mu, var = self.running_mean, self.running_var

        # 标准化：移除均值，再除以标准差
        inv_std = 1.0 / cp.sqrt(var + eps)
        X_norm  = (X - mu) * inv_std
        # 缓存用于反向
        self.cache = (X_norm, inv_std, X.shape, dims)

        # 应用缩放和平移
        if self.init['optimizable']:
            return self.params['gamma'] * X_norm + self.params['beta']
        return X_norm

    def backward(self, grad):
        """
        计算 gamma、beta、和输入 X 的梯度：
          - gamma 梯度: grad * X_norm
          - beta  梯度: grad
          - X 梯度: 根据链式法则处理归一化部分
        """
        X_norm, inv_std, shape, dims = self.cache
        M = cp.prod(cp.array([shape[d] for d in dims]))

        if self.init['optimizable']:
            # 参数梯度
            self.grads['gamma'] = cp.sum(grad * X_norm, axis=dims, keepdims=True)
            self.grads['beta']  = cp.sum(grad, axis=dims, keepdims=True)
            grad_norm = grad * self.params['gamma']
        else:
            grad_norm = grad

        # 输入梯度: 平衡减去均值和规范化项
        dY   = cp.sum(grad_norm, axis=dims, keepdims=True)
        dYX  = cp.sum(grad_norm * X_norm, axis=dims, keepdims=True)
        dX   = (grad_norm - dY / M - X_norm * dYX / M) * inv_std
        return dX

    def clear_grad(self):
        """清空梯度"""
        self.grads = {}
        if self.init['optimizable']:
            self.grads['gamma'] = None
            self.grads['beta']  = None

    def set_params(self, params):
        """手动设置 gamma 和 beta"""
        for k, v in params.items():
            self.params[k] = cp.array(v)

    def set_training(self, train: bool):
        """切换训练/测试模式"""
        self.init['train'] = train
