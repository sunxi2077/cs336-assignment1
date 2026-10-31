"""Small BPE training experiment. Run from the assignment root with -m."""

import argparse
import cProfile
import json
import pickle
import resource
import sys
import tempfile
import time
from pathlib import Path

from cs336_basics.bpe_tokenizer import train_bpe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--max-docs", type=int, default=1000, help="0 uses the entire input file")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/bpe_debug"))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Pre-tokenization processes; default keeps serial behavior")
    args = parser.parse_args()
    special_token = "<|endoftext|>"
    if args.max_docs < 0 or args.vocab_size < 257 or args.workers < 1:
        parser.error("--max-docs must be nonnegative; --vocab-size >= 257; --workers >= 1")
    if not args.input.is_file():
        parser.error(f"Input file not found: {args.input}")
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new --output-dir to preserve previous runs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cs336-bpe-") as temporary_dir:
        training_path = args.input
        document_count = None
        if args.max_docs:
            training_path = Path(temporary_dir) / "subset.txt"
            document_count = 0
            # Stream complete documents without loading the full source file.
            with args.input.open(encoding="utf-8") as source, training_path.open("w", encoding="utf-8") as target:
                finished = False
                for line in source:
                    pieces = line.split(special_token)
                    for piece in pieces[:-1]:
                        target.write(piece)
                        target.write(special_token)
                        document_count += 1
                        if document_count >= args.max_docs:
                            finished = True
                            break
                    if finished:
                        break
                    target.write(pieces[-1])

        input_bytes = training_path.stat().st_size
        print(f"Input: {args.input}; subset bytes: {input_bytes:,}; target vocabulary: {args.vocab_size}", flush=True)
        profiler = cProfile.Profile() if args.profile else None
        start = time.perf_counter()
        if profiler:
            profiler.enable()
        try:
            vocab, merges = train_bpe(training_path, args.vocab_size, [special_token], num_workers=args.workers)
        finally:
            elapsed = time.perf_counter() - start
            if profiler:
                profiler.disable()
                profiler.dump_stats(str(output_dir / "profile.prof"))

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
    ordinary_vocab = {i: token for i, token in vocab.items() if token != special_token.encode("utf-8")}
    longest_id, longest = max(ordinary_vocab.items(), key=lambda entry: len(entry[1]))
    with (output_dir / "vocab.pkl").open("wb") as file:
        pickle.dump(vocab, file)
    with (output_dir / "merges.pkl").open("wb") as file:
        pickle.dump(merges, file)
    # Verify that serialization preserves the actual Python objects.
    with (output_dir / "vocab.pkl").open("rb") as file:
        assert pickle.load(file) == vocab
    with (output_dir / "merges.pkl").open("rb") as file:
        assert pickle.load(file) == merges

    summary = {
        "input": str(args.input.resolve()),
        "max_docs": args.max_docs,
        "document_boundaries_in_subset": document_count,
        "training_input_bytes": input_bytes,
        "target_vocab_size": args.vocab_size,
        "vocab_size": len(vocab),
        "merge_count": len(merges),
        "training_seconds": elapsed,
        "profile_enabled": args.profile,
        "pretokenization_workers": args.workers,
        "memory_scope": "parent process only; excludes workers" if args.workers > 1 else "single process",
        "profile_scope": "parent process only; excludes worker execution" if args.workers > 1 else "single process",
        "process_peak_rss_mib": peak_bytes / 1024**2,
        "longest_non_special_token_id": longest_id,
        "longest_non_special_token_bytes": len(longest),
        "longest_non_special_token_repr": repr(longest),
        "longest_non_special_token_text": longest.decode("utf-8", errors="replace"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved to {output_dir}. Only load pickle files you trust.")


if __name__ == "__main__":
    main()
