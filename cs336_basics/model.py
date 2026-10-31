"""CS336 Assignment 1 —— 模型基础组件。

目前包含 Linear 的起步骨架。其余组件（Embedding / RMSNorm / SwiGLU 等）
按作业进度继续往这里加。
"""

import math

import torch
import torch.nn as nn
from einops import einsum, rearrange


class Linear(nn.Module):
    """线性变换层：y = x W^T。

    注意：按现代 LLM（PaLM / LLaMA）的约定，本作业的 Linear **不带 bias**。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        # super().__init__() 调用父类 torch.nn.Module 的构造函数。
        # nn.Module 在被初始化时会建立"参数注册表"等内部机制；
        # 不调用它，self.weight 不会被识别成模型参数，后面会出错。
        super().__init__()

        # 把维度存成对象属性。self.xxx 表示"挂在这个对象上的属性"，
        # 之后在 forward 里仍能访问到。
        self.in_features = in_features
        self.out_features = out_features

        # torch.empty(shape) 只分配内存、不初始化内容（里面是任意值）。
        # 这里形状写成 (out_features, in_features)，因为后面要算 x @ weight.T：
        #   x 的最后一维是 in_features，
        #   weight.T 的第一维是 in_features —— 两边对得上。
        # device / dtype 直接透传；传 None 时交给 PyTorch 用默认值。
        # nn.Parameter(...) 把普通张量"注册"为可训练参数，优化器才能对它求梯度。
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # 初始化（作业 3.3.1）：权重 ~ N(0, sigma^2)，截断到 [-3sigma, 3sigma]，
        # 其中 sigma^2 = 2 / (in_features + out_features)。
        # trunc_normal_ 带下划线 = "原地修改"传入的张量（不返回新张量），
        # 它改的就是上面的 self.weight。
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (..., in_features) -> 输出 (..., out_features)。

        einsum 模式 "... d_in, d_out d_in -> ... d_out" 就是批量矩阵乘法：
        - d_in 在两边同名 → 缩并（相乘求和）；
        - "..." 表示任意数量的前导批量维，原样保留；
        - d_out 从 weight 带出，成为输出最后一维。
        等价于 x @ self.weight.T，但各轴的含义直接写在代码里（自文档化）。
        """
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """词嵌入层：把 token ID 映射为 d_model 维向量。

    内部是一张 (num_embeddings, embedding_dim) 的查表矩阵，每行对应一个 token。
    forward 用整数 ID 去这张表里"取行"。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 查表矩阵：行 = token，列 = 特征维（d_model 在最后一维，符合嵌入表惯例）
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        # 初始化（作业 3.3.1）：Embedding ~ N(0, 1)，截断到 [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """输入整数 ID (...,) -> 输出 (..., embedding_dim)。"""
        # 用整数张量做下标，逐元素取出对应的行（查表）
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """均方根层归一化（RMSNorm）：

        RMSNorm(a_i) = (a_i / RMS(a)) * g_i
        RMS(a) = sqrt( (1/d_model) * sum_i a_i^2 + eps )

    其中 g 是可学习的增益，初始化为 1。
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # 可学习增益 g，形状 (d_model,)，初始化为全 1（初始不额外缩放）
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (..., d_model) -> 输出 (..., d_model)。"""
        # 记录原始 dtype，算完再转回去
        in_dtype = x.dtype

        # 上转换到 float32，防止平方时溢出
        x = x.to(torch.float32)

        # RMS(a) = sqrt(mean(a^2) + eps)，沿最后一维（特征维）计算
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # 归一化后再乘增益 g（g 沿最后一维广播）
        result = (x / rms) * self.weight

        # 转回原始 dtype 返回
        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU（又名 Swish）激活函数：SiLU(x) = x * sigmoid(x)。"""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络：

        FFN(x) = W2( SiLU(W1 x) ⊙ (W3 x) )

    其中 W1, W3 形状 (d_ff, d_model)，W2 形状 (d_model, d_ff)，均无 bias。
    d_ff 默认取 (8/3)*d_model 并四舍五入到 64 的倍数。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_ff is None:
            # (8/3)*d_model，取整到最近的 64 的倍数（硬件效率）
            d_ff = 64 * round(8 * d_model / 3 / 64)

        self.d_model = d_model
        self.d_ff = d_ff

        # 三个无 bias 的线性层：W1、W3 上投影，W2 下投影
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (..., d_model) -> 输出 (..., d_model)。"""
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    """旋转位置嵌入（RoPE）。

    对输入向量的每一对维度 (x, y)，按 token 位置 i 做二维旋转：

        x' = x·cos θ - y·sin θ
        y' = x·sin θ + y·cos θ
        θ_{i,k} = i · Θ^{-(2k-2)/d_k}

    cos/sin 在 __init__ 中预计算并存成 buffer（不参与训练，无参数）。
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert d_k % 2 == 0, "RoPE 要求 d_k 为偶数（两两配对）"
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 不同维度对的"频率"：Θ^{-(2k-2)/d_k}，k = 0..d_k/2-1
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k)
        )  # (d_k/2,)

        # 每个位置的旋转角：positions × inv_freq  ->  (max_seq_len, d_k/2)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)

        # 预计算 cos/sin 并注册成 buffer（persistent=False，不写入 state_dict）
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """输入 x: (..., seq, d_k)，token_positions: (..., seq) -> 同形状输出。"""
        # 按位置取出对应的 cos/sin： (..., seq, d_k/2)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # 两两配对：偶数索引是每对的第一个，奇数索引是第二个
        x1 = x[..., 0::2]  # (..., seq, d_k/2)
        x2 = x[..., 1::2]  # (..., seq, d_k/2)

        # 应用二维旋转
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # 交错拼回 (..., seq, d_k)
        return torch.stack([out1, out2], dim=-1).flatten(-2)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """沿指定维度做数值稳定的 softmax：

        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    先减去该维最大值，避免 exp 溢出。
    """
    x_max = x.max(dim=dim, keepdim=True).values   # (..., 1, ...) 便于广播
    exp_x = torch.exp(x - x_max)                  # 指数输入 <= 0，不会溢出
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """缩放点积注意力：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V。

    Q: (..., q, d_k)   K: (..., k, d_k)   V: (..., k, d_v)
    mask: (..., q, k) 布尔矩阵，True 表示允许关注（参与 softmax）。
    返回: (..., q, d_v)
    """
    d_k = Q.shape[-1]

    # 分数：Q·K^T，形状 (..., q, k)
    scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)

    # 掩码：在 softmax 之前，把不允许的位置设为 -inf
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # 沿 key 维做 softmax，得到注意力权重 (..., q, k)
    attn = softmax(scores, dim=-1)

    # 按权重对 V 加权求和，得到 (..., q, d_v)
    return einsum(attn, V, "... q k, ... k d_v -> ... q d_v")


class MultiHeadSelfAttention(nn.Module):
    """因果多头自注意力。

    流程：投影出 Q/K/V → 拆成 h 个头 →（可选）对 Q/K 应用 RoPE
    → 每个头独立做带因果掩码的缩放点积注意力 → concat 各头 → W_O 投影。

    所有投影都是无 bias 的 Linear；d_model 必须能被 num_heads 整除。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q/K/V 投影 + 输出投影（均无 bias）
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 可选 RoPE；嵌入维度必须等于每个头的维度 d_k
        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta, self.d_k, max_seq_len, device=device
            )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """输入 x: (..., seq, d_model) -> 输出 (..., seq, d_model)。"""
        seq_len = x.shape[-2]

        # 1) 三次矩阵乘，一次算出所有头的 Q/K/V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2) 拆头： (..., s, d_model) -> (..., h, s, d_k)
        Q = rearrange(Q, "... s (h dk) -> ... h s dk", h=self.num_heads)
        K = rearrange(K, "... s (h dk) -> ... h s dk", h=self.num_heads)
        V = rearrange(V, "... s (h dk) -> ... h s dk", h=self.num_heads)

        # 3) RoPE 只作用于 Q、K（每个头独立，head 维作为批量维）
        if self.rope is not None and token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # 4) 因果掩码：下三角（含对角线）为 True，允许看自己及之前
        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        # 5) 每个头独立做缩放点积注意力
        attn = scaled_dot_product_attention(Q, K, V, mask=mask)

        # 6) 合头： (..., h, s, d_k) -> (..., s, d_model)
        attn = rearrange(attn, "... h s dk -> ... s (h dk)")

        # 7) 输出投影（跨头混合）
        return self.output_proj(attn)


class TransformerBlock(nn.Module):
    """pre-norm Transformer 块：

        y = x + MultiHeadSelfAttention(RMSNorm(x))
        z = y + SwiGLU(RMSNorm(y))

    每个子层都是"先 RMSNorm，再做子层运算，最后残差相加"。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # RoPE 需要位置；没给就默认 0..seq-1
        if token_positions is None and self.attn.rope is not None:
            token_positions = torch.arange(x.shape[-2], device=x.device)

        # 第一子层：注意力 + 残差
        y = x + self.attn(self.ln1(x), token_positions)
        # 第二子层：前馈 + 残差
        z = y + self.ffn(self.ln2(y))
        return z


class TransformerLM(nn.Module):
    """完整的 Transformer 语言模型：

        token IDs -> Embedding -> num_layers 个 TransformerBlock
                  -> 最终 RMSNorm -> LM head -> logits
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )

        # 必须用 nn.ModuleList（普通 list 不会注册子模块，参数会丢）
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

        # 记录上下文长度，生成时用于裁剪输入
        self.context_length = context_length

    def forward(
        self,
        token_ids: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """输入 token_ids: (batch, seq) -> logits: (batch, seq, vocab_size)。"""
        x = self.token_embeddings(token_ids)  # (batch, seq, d_model)

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.ln_final(x)
        return self.lm_head(x)  # (batch, seq, vocab_size)


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """交叉熵损失（对 inputs 的最后一维做 softmax，再对 targets 取 -log）。

    inputs: (..., V) 的 logits；targets: (...) 的整数 token ID。
    返回标量：所有位置的 -log p 的平均。

    数值稳定：先减去最后一维的最大值，并直接使用 logits（抵消 log/exp）：
        loss = logsumexp(o - m) - (o[y] - m)
    """
    # 1) 沿最后一维减去最大值
    m = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - m

    # 2) log(sum(exp(o - m)))
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))  # (...)

    # 3) 取出正确 token 的 logit（直接用 shifted，不重复 exp/log）
    target_logits = torch.gather(
        shifted, -1, targets.unsqueeze(-1)
    ).squeeze(-1)  # (...)

    # 4) loss = log_sum_exp - target_logits，再对所有位置取平均
    return (log_sum_exp - target_logits).mean()
