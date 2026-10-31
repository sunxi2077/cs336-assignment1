"""实验 2.7(a)：固定取验证集前十篇非空文档，测量 bytes/token。"""

import json
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


def first_documents(path: Path, count: int) -> list[str]:
    # 分块读取并保留未读完整的尾部，标记跨块也不会丢失。
    # 分隔符不计入样本文本；保留文档本身的全部空白，不使用 strip 改写。
    marker = "<|endoftext|>"
    documents = []
    pending = ""
    with path.open(encoding="utf-8", newline="") as file:
        while len(documents) < count:
            chunk = file.read(65536)
            if not chunk:
                if pending.strip():
                    documents.append(pending)
                break
            pending += chunk
            pieces = pending.split(marker)
            pending = pieces.pop()
            for document in pieces:
                if document.strip():
                    documents.append(document)
                    if len(documents) == count:
                        return documents
    if len(documents) != count:
        raise ValueError(f"Expected {count} documents in {path}, found {len(documents)}")
    return documents


def main() -> None:
    configurations = [
        ("TinyStories", "data/TinyStoriesV2-GPT4-valid.txt", "artifacts/bpe_tinystories_parallel4", 10000),
        ("OpenWebText", "data/owt_valid.txt", "artifacts/bpe_owt_full_parallel2_vocab32k", 32000),
    ]
    results = []
    for name, source, trained_dir, expected_size in configurations:
        directory = Path(trained_dir)
        tokenizer = Tokenizer.from_files(
            str(directory / "vocab.pkl"), str(directory / "merges.pkl"),
            special_tokens=["<|endoftext|>"],
        )
        assert len(tokenizer.vocab) == expected_size
        documents = first_documents(Path(source), 10)
        records = []
        total_bytes = 0
        total_tokens = 0
        for document in documents:
            ids = tokenizer.encode(document)
            assert tokenizer.decode(ids) == document, "Encoding roundtrip failed"
            byte_count = len(document.encode("utf-8"))
            total_bytes += byte_count
            total_tokens += len(ids)
            records.append({"text": document, "ids": ids,
                            "utf8_bytes": byte_count, "tokens": len(ids)})
        summary = {
            "dataset": name,
            "source": source,
            "tokenizer_directory": trained_dir,
            "vocab_size": len(tokenizer.vocab),
            "sampling": "first 10 non-whitespace documents; not a random sample",
            "boundary_marker_included": False,
            "document_count": len(documents),
            "total_utf8_bytes": total_bytes,
            "total_tokens": total_tokens,
            # 先求总字节、总 token，再相除；不是逐篇比值的算术平均。
            "bytes_per_token": total_bytes / total_tokens,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        results.append({"summary": summary, "documents": records})

    output = Path("artifacts/tokenizer_compression_first10.json")
    # x 模式防止覆盖已有实验数据。
    with output.open("x", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Saved samples, IDs and measurements to {output}")


if __name__ == "__main__":
    main()
