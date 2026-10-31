"""CS336 Assignment 1 —— 文本生成（第 6 章 decoding）。

从 TransformerLM 逐步采样生成 token，支持：
  - 温度缩放（temperature）
  - top-p / 核采样（nucleus sampling）
  - 遇到 <|endoftext|> 停止
"""

import torch

from cs336_basics.model import softmax


def top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    """top-p（核采样）：只保留"累计概率达到 p 的最小 token 集合"，再归一化。

    probs: (..., V) 概率分布，沿最后一维处理。
    返回同形状、过滤并重新归一化后的分布。
    """
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # 保留"在加入它之前累计和仍 < p"的 token（即包含跨过 p 的那个）
    keep = (cumulative - sorted_probs) < p

    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # 散射回原来的 token 顺序
    out = torch.zeros_like(probs)
    out.scatter_(-1, sorted_idx, sorted_probs)
    return out


@torch.no_grad()
def generate(
    model,
    prompt_ids,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    eot_token_id: int | None = None,
    context_length: int | None = None,
):
    """从模型采样生成 token 序列。

    Args:
        model: TransformerLM。
        prompt_ids: (T,) 或 (1, T) 的整数 token ID。
        max_new_tokens: 最多生成多少个 token。
        temperature: 温度；<= 0 表示贪心（argmax）。
        top_p: 核采样阈值；None 表示不做 top-p。
        eot_token_id: 结束 token 的 ID，生成到它就停。
        context_length: 输入裁剪的最大长度；默认取 model.context_length。

    Returns:
        (1, T + 生成数) 的 IntegerLongTensor，包含 prompt。
    """
    device = next(model.parameters()).device
    ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)

    if context_length is None:
        context_length = getattr(model, "context_length", None)

    model.eval()
    for _ in range(max_new_tokens):
        inp = ids if context_length is None else ids[:, -context_length:]
        logits = model(inp)[:, -1, :]  # (1, V) 最后一个位置的下一个 token 分数

        if temperature == 0:
            # 贪心：直接取最大
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = softmax(logits / temperature, dim=-1)
            if top_p is not None:
                probs = top_p_filter(probs, top_p)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        ids = torch.cat([ids, next_id], dim=-1)

        if eot_token_id is not None and next_id.item() == eot_token_id:
            break

    return ids
