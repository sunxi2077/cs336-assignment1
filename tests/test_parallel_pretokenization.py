from cs336_basics.bpe_tokenizer import (
    pretokenization, split_special_token, pretokenize_file_parallel, train_bpe,
)


def test_parallel_counts_and_training(tmp_path):
    marker = "<|endoftext|>"
    path = tmp_path / "corpus.txt"
    path.write_text(marker + (("猫🙂 hello  world\n" + marker) * 25) + "尾巴", encoding="utf-8")
    expected = pretokenization(split_special_token(path, [marker]))
    assert pretokenize_file_parallel(path, [marker], 2, chunk_bytes=17) == expected
    assert train_bpe(path, 290, [marker], num_workers=2) == train_bpe(path, 290, [marker])


def test_marker_spans_search_window(tmp_path):
    marker = "<|endoftext|>"
    path = tmp_path / "corpus.txt"
    path.write_text("x" * 4105 + marker + "猫" + marker + "end", encoding="utf-8")
    expected = pretokenization(split_special_token(path, [marker]))
    assert pretokenize_file_parallel(path, [marker], 2, chunk_bytes=10) == expected


def test_no_marker_and_multiple_specials(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("猫🙂 a<|a|>b<|ab|>c", encoding="utf-8")
    for specials in (["<|endoftext|>"], [], ["<|a|>", "<|ab|>"]):
        expected = pretokenization(split_special_token(path, specials))
        assert pretokenize_file_parallel(path, specials, 2, chunk_bytes=5) == expected
