"""流式编码数据集，保存为可用 NumPy memmap 读取的 uint16 原始数组。"""

import argparse
import hashlib
import json
import time
from itertools import islice
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=0, help="0 encodes the whole file; positive values are debug samples")
    args = parser.parse_args()
    if args.max_chars < 0:
        parser.error("--max-chars must be nonnegative")
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    metadata = args.output.with_suffix(args.output.suffix + ".json")
    if any(path.exists() for path in (args.output, partial, metadata)):
        parser.error("Output/partial/metadata exists; choose a new output path")
    tokenizer = Tokenizer.from_files(
        str(args.tokenizer_dir / "vocab.pkl"), str(args.tokenizer_dir / "merges.pkl"),
        ["<|endoftext|>"],
    )
    if not tokenizer.vocab or min(tokenizer.vocab) < 0 or max(tokenizer.vocab) > 65535:
        parser.error("Token IDs do not fit uint16")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    restored_hash = hashlib.sha256()
    source_bytes = 0

    def text_chunks():
        nonlocal source_bytes
        remaining = args.max_chars
        # newline='' 保留原文件的换行符，不将 CRLF 转换为 LF。
        with args.input.open(encoding="utf-8", newline="") as source:
            while True:
                size = min(65536, remaining) if args.max_chars else 65536
                if size == 0:
                    break
                text = source.read(size)
                if not text:
                    break
                encoded = text.encode("utf-8")
                source_bytes += len(encoded)
                source_hash.update(encoded)
                yield text
                if args.max_chars:
                    remaining -= len(text)

    start = time.perf_counter()
    last_report = start
    token_count = 0
    ids = tokenizer.encode_iterable(text_chunks())
    # 固定大小批次写盘；不创建整个数据集的 Python ID 列表。
    with partial.open("xb") as destination:
        while True:
            batch = np.fromiter(islice(ids, 65536), dtype="<u2")
            if not batch.size:
                break
            batch.tofile(destination)
            token_count += int(batch.size)
            # 比较还原 bytes 的哈希，批次可以位于 UTF-8 字符中间。
            restored_hash.update(b"".join(tokenizer.vocab[int(i)] for i in batch))
            now = time.perf_counter()
            if now - last_report >= 10:
                print(f"Read {source_bytes / 1024**2:.1f} MiB; wrote {token_count:,} IDs; elapsed {now-start:.1f}s", flush=True)
                last_report = now
    elapsed = time.perf_counter() - start
    if source_hash.digest() != restored_hash.digest():
        raise RuntimeError("Roundtrip byte hash mismatch; partial file retained for inspection")
    if partial.stat().st_size != token_count * 2:
        raise RuntimeError("Unexpected output size")
    partial.rename(args.output)
    summary = {
        "input": str(args.input.resolve()), "tokenizer_dir": str(args.tokenizer_dir.resolve()),
        "max_chars": args.max_chars, "source_utf8_bytes": source_bytes,
        "tokens": token_count, "output_bytes": token_count * 2,
        "dtype": "<u2", "format": "raw NumPy array; use np.memmap(path, dtype='<u2', mode='r')",
        "seconds_including_write_and_verification": elapsed,
        "bytes_per_second": source_bytes / elapsed,
        "bytes_per_token": source_bytes / token_count if token_count else None,
        "roundtrip_sha256": source_hash.hexdigest(),
        "special_tokens_preserved": True,
    }
    with metadata.open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
