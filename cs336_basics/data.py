"""CS336 Assignment 1 —— 数据加载。

把拼接好的一维 token 序列切成 (输入, 下一个 token) 的批量。
"""

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从一维 token 数组中采样一个 batch。

    Args:
        dataset: 一维整数数组（可以是 np.memmap）。
        batch_size: 批大小。
        context_length: 每条序列的长度 m。
        device: PyTorch 设备字符串，如 "cpu" / "cuda:0" / "mps"。

    Returns:
        (x, y)，形状都是 (batch_size, context_length) 的 LongTensor：
          x[b] = dataset[i : i+m]
          y[b] = dataset[i+1 : i+m+1]   （即 x 右移一位）
    """
    n = len(dataset)

    # 合法的起始下标：0 .. n-m-1（共 n-m 个），保证 y 不越界
    starts = np.random.randint(0, n - context_length, size=batch_size)

    # 用广播构造所有下标： (batch_size, context_length)
    offsets = np.arange(context_length)
    idx = starts[:, None] + offsets[None, :]

    x = dataset[idx]
    y = dataset[idx + 1]

    return (
        torch.tensor(x, dtype=torch.long, device=device),
        torch.tensor(y, dtype=torch.long, device=device),
    )
