"""CS336 Assignment 1 —— 优化器。

实现 AdamW（Adam + 解耦权重衰减），对应讲义算法 1。
"""

import math
from typing import Callable, Optional

import torch


class AdamW(torch.optim.Optimizer):
    """AdamW 优化器。

    每个参数维护：
      - m：一阶矩（梯度的指数滑动平均）
      - v：二阶矩（梯度平方的指数滑动平均）

    更新流程（讲义算法 1）：
      1. 折入偏差修正的步长 alpha_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
      2. 解耦权重衰减：theta <- theta - lr * wd * theta
      3. m <- beta1*m + (1-beta1)*g
      4. v <- beta2*v + (1-beta2)*g^2
      5. theta <- theta - alpha_t * m / (sqrt(v) + eps)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # 初始化每个参数的状态
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                t = state["t"] + 1
                m = state["m"]
                v = state["v"]

                # 1) 折入偏差修正的步长
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # 2) 解耦权重衰减
                p.data.mul_(1 - lr * weight_decay)

                # 3) 一阶矩
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 4) 二阶矩
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 5) 自适应更新
                p.data -= alpha_t * m / (v.sqrt() + eps)

                state["t"] = t

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """带线性预热的余弦退火学习率调度（讲义 4.4）。

    - t < T_w：线性预热      alpha_t = (t / T_w) * alpha_max
    - T_w <= t <= T_c：余弦   alpha_t = alpha_min + 0.5*(1+cos(pi*ratio))*(alpha_max-alpha_min)
    - t > T_c：固定在 alpha_min
    """
    if it < warmup_iters:
        # 预热：从 0 线性升到 alpha_max
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        # 余弦退火：从 alpha_max 平滑降到 alpha_min
        ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (
            1 + math.cos(math.pi * ratio)
        ) * (max_learning_rate - min_learning_rate)
    else:
        # 退火结束后保持最小学习率
        return min_learning_rate


def gradient_clipping(parameters, max_l2_norm: float) -> None:
    """把一组参数的梯度按总 L2 范数裁剪到不超过 max_l2_norm（就地修改）。

    计算所有梯度的总 L2 范数 ||g||_2；若超过 max_l2_norm，则
    对所有梯度乘以 max_l2_norm / (||g||_2 + eps)，否则不变。
    """
    params = [p for p in parameters if p.grad is not None]

    # 总 L2 范数：sqrt(sum_p ||g_p||^2)
    total_norm = torch.sqrt(sum((p.grad**2).sum() for p in params))

    # 裁剪系数；clip 到 <= 1，保证只在"太大"时才缩小
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1.0)

    for p in params:
        p.grad.mul_(clip_coef)
