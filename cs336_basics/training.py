"""CS336 Assignment 1 —— 训练基础设施：checkpoint 保存/加载。"""

import torch


def save_checkpoint(model, optimizer, iteration: int, out) -> None:
    """把模型、优化器状态和当前迭代数保存到 out（路径或文件对象）。"""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(src, model, optimizer) -> int:
    """从 src 恢复模型和优化器状态，返回之前保存的迭代数。"""
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
