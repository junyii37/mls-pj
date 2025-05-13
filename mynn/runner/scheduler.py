import cupy as cp


class CosineAnnealingLR:
    """
        余弦退火调度策略

        Args:
            optimizer (Optimizer): 优化器
            T_max (int): 周期
            eta_min (float): 最低学习率
            last_epoch (int): 最新的 epoch
    """
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.lr_max = optimizer.lr  # 设定初始学习率为 lr_max


    def step(self):
        self.last_epoch += 1
        lr = self.eta_min + 0.5 * (self.lr_max - self.eta_min) * (1 + cp.cos(cp.pi * self.last_epoch / self.T_max))
        self.optimizer.lr = lr


class StepLR:
    """
    分段下降调度策略

    Args:
        optimizer: 优化器，需具有属性 `lr`
        step_size (int): 每隔多少个 epoch 下降一次
        gamma (float): 学习率下降倍数
        last_epoch (int): 已经执行过的 epoch 数，默认为 -1，表示还没开始
    """
    def __init__(self, optimizer, step_size=30, gamma=0.1, last_epoch=-1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        # 记录初始学习率
        self.base_lr = optimizer.lr

    def step(self):
        self.last_epoch += 1
        # 计算已经跨过了多少个 step_size
        n = self.last_epoch // self.step_size
        # 更新 lr
        lr = self.base_lr * (self.gamma ** n)
        self.optimizer.lr = lr


class MultiStepLR:
    """
    多阶段下降调度策略

    Args:
        optimizer: 优化器，需具有属性 `lr`
        milestones (list of int): 在哪些 epoch 点进行下降
        gamma (float): 每次下降的倍数
        last_epoch (int): 已执行的 epoch 数，默认为 -1
    """
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.optimizer = optimizer
        self.milestones = set(milestones)
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.base_lr = optimizer.lr

    def step(self):
        self.last_epoch += 1
        # 统计当前 epoch 应当下降了多少次
        k = sum(1 for m in self.milestones if self.last_epoch >= m)
        lr = self.base_lr * (self.gamma ** k)
        self.optimizer.lr = lr


class ExponentialLR:
    """
    指数衰减调度策略

    Args:
        optimizer: 优化器，需具有属性 `lr`
        gamma (float): 每个 epoch 学习率乘以该倍数
        last_epoch (int): 已执行的 epoch 数，默认为 -1
    """
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.optimizer = optimizer
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.base_lr = optimizer.lr

    def step(self):
        self.last_epoch += 1
        lr = self.base_lr * (self.gamma ** self.last_epoch)
        self.optimizer.lr = lr