class Optimizer:
    """
        Basic optimizer.

        基类优化器。

        Args:
            lr (float): 学习率。
    """
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        raise NotImplementedError

