from ..layer import Layer
from ..activation import ReLU
from ..regularization import BN
from .. import Conv
from ..initialization import He



class BasicBlock(Layer):
    def __init__(self, in_channels, out_channels, downsampling=False, initialize_method=He, weight_decay=0.2, optimizable=True):
        super().__init__()
        # 基块配置参数
        self.init = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'downsampling': downsampling,
            'initialize_method': initialize_method,
            'weight_decay': weight_decay,
            'optimizable': optimizable
        }
        # BasicBlock 本身无直接可优化参数，交由子层管理
        self.params = {}
        # 子层初始化
        # 主分支
        self.conv1 = Conv(
            in_channel=in_channels,
            out_channel=out_channels,
            kernel=4 if downsampling else 3,
            stride=2 if downsampling else 1,
            padding=1,
            initialize_method=initialize_method,
            weight_decay=weight_decay,
            optimizable=optimizable
        )
        self.bn1 = BN(
            normalized_dims=(0, 2, 3),
            param_shape=(1, out_channels, 1, 1)
        )
        self.relu1 = ReLU()
        self.conv2 = Conv(
            in_channel=out_channels,
            out_channel=out_channels,
            kernel=3,
            stride=1,
            padding=1,
            initialize_method=initialize_method,
            weight_decay=weight_decay,
            optimizable=optimizable
        )
        self.bn2 = BN(
            normalized_dims=(0, 2, 3),
            param_shape=(1, out_channels, 1, 1)
        )
        # Shortcut 分支
        if downsampling:
            self.short_conv = Conv(
                in_channel=in_channels,
                out_channel=out_channels,
                kernel=2,
                stride=2,
                padding=0,
                initialize_method=initialize_method,
                weight_decay=weight_decay,
                optimizable=optimizable
            )
            self.short_bn = BN(
                normalized_dims=(0, 2, 3),
                param_shape=(1, out_channels, 1, 1)
            )
        else:
            self.short_conv = None
            self.short_bn = None
        # 最后激活
        self.relu2 = ReLU()

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # 主分支
        out = self.conv1.forward(X)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        # Shortcut
        if self.short_conv is not None:
            sc = self.short_conv.forward(X)
            sc = self.short_bn.forward(sc)
        else:
            sc = X
        # 相加并激活
        out = out + sc
        out = self.relu2.forward(out)
        # 缓存用于 backward
        self.cache = (X, sc)
        return out

    def backward(self, grad_out):
        X, sc = self.cache
        # ReLU2
        grad = self.relu2.backward(grad_out)
        # 分支梯度
        grad_main, grad_sc = grad, grad
        # Shortcut path
        if self.short_conv is not None:
            grad_sc = self.short_bn.backward(grad_sc)
            grad_sc = self.short_conv.backward(grad_sc)
        # 主分支
        grad = self.bn2.backward(grad_main)
        grad = self.conv2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.bn1.backward(grad)
        grad = self.conv1.backward(grad)
        # 合并梯度
        return grad + grad_sc

    def clear_grad(self):
        """清空子层梯度"""
        for layer in [self.conv1, self.bn1, self.conv2, self.bn2,
                      self.short_conv, self.short_bn]:
            if layer is not None and hasattr(layer, 'clear_grad'):
                layer.clear_grad()

    def set_params(self, params):
        for layer, param in zip([self.conv1, self.bn1, self.conv2, self.bn2,
                      self.short_conv, self.short_bn], params):
            if layer is not None and hasattr(layer, 'set_params'):
                layer.set_params(param)
