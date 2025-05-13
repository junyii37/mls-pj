import cupy as cp


class EarlyStopping:
    """
        早停策略

        Args:
            patience (int): 耐心。允许在验证集表现没有提高的情况下继续训练的轮数
            delta (float): 容忍度。如果验证集的精度提升小于 delta，则认为没有进展
    """
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_accuracy = -cp.inf  # 初始化为负无穷大
        self.counter = 0

    def step(self, val_accuracy):
        if val_accuracy > self.best_accuracy + self.delta:
            self.best_accuracy = val_accuracy
            self.counter = 0
            print(f"Patience Counter Reset to {self.counter}\n")
        else:
            self.counter += 1
            print(f"Patience: {self.counter}/{self.patience}\n")
            if self.counter >= self.patience:
                print(f"Early Stopping Triggered.\n")
                return True
        return False
