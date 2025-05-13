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


class CrossEntropy:
    """
    交叉熵损失层，将 softmax 与 cross‐entropy 合并
    数值稳定：forward 中调用 above softmax
    反向：直接用 (p - y)/N
    """
    def __init__(self, model=None):
        self.model = model  # 用于把梯度传给前面的网络

    def __call__(self, X, Y):
        return self.forward(X, Y)

    def forward(self, X, Y):
        """
        计算 loss 和 accuracy
        输入：
          X: 未经过 softmax 的 logits，shape (batch_size, channel)
          Y: 标签，one-hot 或索引格式
        返回：
          loss: 标量平均交叉熵
          acc: 标量准确率
        """
        # 先做 softmax，得到概率
        Y_pred = softmax(X)
        # 缓存用于反向
        self.cache = (Y_pred, Y)

        # 计算交叉熵损失
        logp = cp.log(Y_pred + 1e-8)  # +eps 保证数值稳定
        if Y.ndim == 2:
            # one-hot 格式
            loss_vec = -cp.sum(Y * logp, axis=1)
        else:
            # 索引格式
            idx = cp.arange(X.shape[0])
            loss_vec = -logp[idx, Y]
        loss = cp.mean(loss_vec)

        # 计算准确率
        acc = accuracy(Y_pred, Y)
        return loss, acc

    def backward(self):
        """
        反向传播梯度到模型
        dL/dX = (Y_pred - Y) / N
        """
        Y_pred, Y = self.cache
        N = Y_pred.shape[0]

        if Y.ndim == 2:
            grad = (Y_pred - Y) / N
        else:
            # 从索引格式构造 grad
            grad = Y_pred.copy()
            idx = cp.arange(N)
            grad[idx, Y] -= 1
            grad /= N

        # 传给模型的 backward
        self.model.backward(grad)
