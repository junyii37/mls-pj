class Layer:
    """ 基类，统一神经网络层的形式 """
    def __init__(self):
        # 可更新参数
        self.params = {}
        # 可优化参数
        self.init = {}
        # 参数梯度
        self.grads = {}

    def forward(self, X):
        raise NotImplementedError

    def backward(self, grads):
        raise NotImplementedError