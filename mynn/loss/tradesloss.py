import cupy as cp

def softmax(X):
    """
    输入： X，形状 (batch_size, channel)
    输出： softmax 概率，形状 (batch_size, channel)
    数值稳定性：先减去每行最大值，再做指数
    """
    # 减去每行最大值，防止 exp 数值溢出
    x_max = cp.max(X, axis=1, keepdims=True)
    e = cp.exp(X - x_max)
    # 归一化
    return e / cp.sum(e, axis=1, keepdims=True)


def accuracy(Y_pred, Y):
    """
    计算预测准确率
    输入：
      Y_pred: 预测概率，shape (batch_size, channel)
      Y: 标签，可为 one-hot，shape (batch_size, channel)；也可为索引，shape (batch_size,)
    输出：标量准确率
    """
    pred_labels = cp.argmax(Y_pred, axis=1)
    # 支持 one-hot 或索引两种格式
    if Y.ndim == 2:
        true_labels = cp.argmax(Y, axis=1)
    else:
        true_labels = Y
    return cp.mean(pred_labels == true_labels)


class TRADESLoss:
    def __init__(self, model, beta=6.0):
        self.model = model    # 关联的模型（用于反向传播）
        self.beta = beta      # 对抗损失的权重系数
        self.cache = None     # 缓存前向传播的中间结果

    def __call__(self, logits_clean, logits_adv, y):
        return self.forward(logits_clean, logits_adv, y)

    def forward(self, logits_clean, logits_adv, y):
        """
        计算TRADES总损失
        :param logits_clean: 原始样本的输出logits（未经过softmax）
        :param logits_adv: 对抗样本的输出logits
        :param y: 真实标签（one-hot或类别索引）
        :return: 总损失值, 准确率
        """
        # 计算交叉熵损失（原始样本）
        p_clean = softmax(logits_clean)
        ce_loss, acc = self._compute_ce_loss(p_clean, y)

        # 计算KL散度（对抗样本 vs 原始样本）
        p_adv = softmax(logits_adv)
        kl_loss = self._compute_kl(p_clean, p_adv)

        # 总损失
        total_loss = ce_loss + self.beta * kl_loss

        # 缓存反向传播所需数据
        self.cache = (p_clean, p_adv, y)
        return total_loss, acc

    def backward(self):
        p_clean, p_adv, y = self.cache
        B = p_clean.shape[0]

        # 交叉熵梯度（仅针对 clean 部分）
        if y.ndim == 2:
            grad_ce = (p_clean - y) / B
        else:
            grad_ce = p_clean.copy()
            idx = cp.arange(B)
            grad_ce[idx, y] -= 1
            grad_ce /= B

        # KL 散度梯度（针对 adv 部分）
        grad_kl = (p_adv - p_clean) / B * self.beta

        # 合并梯度（clean部分 + adv部分）
        # 注意：clean和adv在合并的输入中是连续存放的
        grad_combined = cp.concatenate([grad_ce, grad_kl], axis=0)

        # 反向传播到模型
        self.model.backward(grad_combined)
    # ---------- 工具函数 ----------
    def _compute_ce_loss(self, p_clean, y):
        """计算交叉熵损失（与CrossEntropy类一致）"""
        if y.ndim == 2:
            # one-hot格式
            loss_vec = -cp.sum(y * cp.log(p_clean + 1e-8), axis=1)
        else:
            # 类别索引格式
            idx = cp.arange(p_clean.shape[0])
            loss_vec = -cp.log(p_clean[idx, y] + 1e-8)
        ce_loss = cp.mean(loss_vec)
        acc = accuracy(p_clean, y)
        return ce_loss, acc

    def _compute_kl(self, p, q):
        """计算KL散度 KL(p || q)"""
        kl = cp.sum(p * (cp.log(p + 1e-8) - cp.log(q + 1e-8)), axis=1)
        return cp.mean(kl)

