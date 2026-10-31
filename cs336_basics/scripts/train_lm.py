"""CS336 Assignment 1 —— 训练脚本（5.3 training_together）。

把数据、模型、损失、优化器、学习率调度、梯度裁剪、checkpoint 串起来。

Mac（低资源）示例：
    uv run python cs336_basics/scripts/train_lm.py \
        --train-path data/tinystories_train.bin \
        --val-path data/tinystories_valid.bin \
        --vocab-size 10000 --context-length 256 \
        --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 \
        --batch-size 32 --max-steps 5000 \
        --lr 1e-3 --warmup-iters 100 \
        --checkpoint-path artifacts/lm_ckpt.pt

服务器（GPU）只需改数据路径；device 会自动选 cuda。
"""

import argparse
import math
import sys
from pathlib import Path

# 让脚本能从仓库根目录 import cs336_basics
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.training import load_checkpoint, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Train a Transformer LM (CS336 A1).")

    # 数据
    p.add_argument("--train-path", required=True, help="训练集 .bin (uint16)")
    p.add_argument("--val-path", required=True, help="验证集 .bin (uint16)")

    # 模型
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--theta", type=float, default=10000.0)

    # 优化
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # 日志 / checkpoint
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--checkpoint-path", type=str, default=None)
    p.add_argument("--resume", action="store_true")

    # 其它
    p.add_argument("--device", type=str, default="auto",
                   help="auto / cpu / cuda / cuda:0 / mps")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device, num_batches):
    model.eval()
    total = 0.0
    for _ in range(num_batches):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        total += cross_entropy(logits, y).item()
    model.train()
    return total / num_batches


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"device = {device}")

    # 数据：内存映射，避免整个读进内存
    train_data = np.memmap(args.train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_path, dtype=np.uint16, mode="r")
    print(f"train tokens = {len(train_data):,} | val tokens = {len(val_data):,}")

    # 模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters = {n_params:,}")

    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # 可选：从 checkpoint 续训
    start_step = 0
    if args.resume and args.checkpoint_path and Path(args.checkpoint_path).exists():
        start_step = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"resumed from step {start_step}")

    # 余弦调度在 max_steps 处降到 min_lr
    cosine_iters = args.max_steps

    for step in range(start_step, args.max_steps):
        # 1) 按调度设置学习率
        lr = get_lr_cosine_schedule(
            step, args.lr, args.min_lr, args.warmup_iters, cosine_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 2) 取数据 + 前向 + 损失
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)

        # 3) 反向 + 裁剪 + 更新
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        # 4) 日志
        if step % args.log_interval == 0:
            print(f"step {step:6d} | lr {lr:.2e} | train loss {loss.item():.4f}")

        # 5) 定期评估
        if step % args.eval_interval == 0:
            val_loss = evaluate(
                model, val_data, args.batch_size, args.context_length,
                device, args.eval_batches,
            )
            print(
                f"step {step:6d} | val loss {val_loss:.4f} | ppl {math.exp(val_loss):.2f}"
            )

        # 6) 定期保存 checkpoint
        if (
            args.checkpoint_path
            and step > 0
            and step % args.checkpoint_interval == 0
        ):
            save_checkpoint(model, optimizer, step, args.checkpoint_path)
            print(f"step {step:6d} | saved checkpoint -> {args.checkpoint_path}")

    # 最后保存一次
    if args.checkpoint_path:
        save_checkpoint(model, optimizer, args.max_steps, args.checkpoint_path)
        print(f"saved final checkpoint -> {args.checkpoint_path}")


if __name__ == "__main__":
    main()
