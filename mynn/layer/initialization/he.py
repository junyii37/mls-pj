import cupy as cp


def He(size):
    """He 初始化可更新参数，适用于 ReLU 激活函数"""
    # 假设 b 的形状为 (out_channel, )
    if len(size) == 1:
        return cp.zeros(size)
    # 假设线性层参数 W 的形状为 (in_channel, out_channel)
    elif len(size) == 2:
        C_in = size[0]
        std = cp.sqrt(2 / C_in)
        return cp.random.normal(0, std, size)
    # 假设卷积层参数 W 的形状为 (out_channel, in_channel, kernel, kernel)
    elif len(size) == 4:
        C_in = size[1]
        std = cp.sqrt(2.0 / C_in)
        return cp.random.normal(0, std, size)
    else:
        raise ValueError(f"参数形状 {size} 设置错误！")