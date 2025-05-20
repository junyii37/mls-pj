from .optimizer import Optimizer
from ..layer.blocks import BasicBlock


class SGD(Optimizer):
    """
        SGD optimizer with Weight Decay Regularization

        带权重衰减的 SGD 优化器，本质上是 SGD 与 L2 正则化的结合。
    """
    def __init__(self, model=None, lr=0.01):
        super().__init__(model=model, lr=lr)

    def step(self):
        for layer in self.model.layers:
            if layer.init['optimizable']:
                if isinstance(layer, BasicBlock):
                    # 对 BasicBlock 的所有子层分别处理
                    for sublayer in [layer.conv1, layer.bn1, layer.conv2, layer.bn2,
                                    layer.short_conv, layer.short_bn]:
                        if sublayer is not None and sublayer.init['optimizable']:
                            for key in sublayer.params.keys():
                                # 权重衰减
                                sublayer.params[key] *= (1 - self.lr * sublayer.init['weight_decay'])
                                # 参数更新
                                sublayer.params[key] -= self.lr * sublayer.grads[key]
                    continue

                for key in layer.params.keys():
                    # 权重衰减
                    layer.params[key] *= (1 - self.lr * layer.init['weight_decay'])
                    # 参数更新
                    layer.params[key] = layer.params[key] - self.lr * layer.grads[key]
