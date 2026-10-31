"""使用已有词表和 merge 顺序进行编码；不在这里训练新规则。

建议阅读顺序：__init__ → decode → _encode_pretoken → _matches
→ _encode_piece → encode → from_files → encode_iterable。
最后一个方法处理任意输入块的边界，暂时跳过也不影响理解普通编码。
"""

# pickle 将 Python 对象保存到二进制文件，也能从文件还原对象。
import pickle
# Iterable[str]：可以逐项遍历、每项为 str 的对象，例如列表或文本文件。
# Iterator[int]：可以逐项取得 int 的迭代器，不必一次构造完整列表。
from collections.abc import Iterable, Iterator
# 缓存函数计算结果的工具；属于性能优化，不属于 BPE 的核心规则。
from functools import lru_cache

import regex

from cs336_basics.bpe_tokenizer import PAT

class Tokenizer:
    # class 定义一类对象。t = Tokenizer(vocab, merges) 会创建一个实例。
    # __init__ 在创建实例时初始化它；self 就是正在初始化的那个实例。
    # 调用 t.encode(text) 时，Python 自动把 t 作为 self 传入。
    # self.xxx 是保存在实例上的属性，可供这个实例的其他方法使用；
    # 普通局部变量只在当前方法执行期间使用。
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        # 复制输入，避免补充 special token 时修改调用者的词表。
        self.vocab = dict(vocab)
        # 字典推导式：遍历 (ID, bytes)，交换位置，建立 bytes -> ID。
        # 例如 {97: b'a'} 对应的反向表就是 {b'a': 97}。
        self.byte_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merges = list(merges)
        # enumerate 给出 (下标, 元素)，下标就是训练顺序，越小越优先。
        # 例如 merges=[(a,b),(ab,c)]，得到 {(a,b):0, (ab,c):1}。
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        # special_tokens or []：传入 None/空列表时使用空列表。
        # dict.fromkeys 按原顺序去重，再用 list 取出这些键。
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        # any(...)：只要有一个元素满足条件就为 True。
        # not token 在 token == '' 时为 True；空标记无法作为有效边界。
        if any(not token for token in self.special_tokens):
            raise ValueError("Special tokens must not be empty")
        # 遍历字典默认遍历键。最大 ID 加 1，比 len(vocab) 更兼容不连续 ID。
        # default=-1 用于空词表，此时新 ID 从 0 开始。
        next_id = max(self.vocab, default=-1) + 1
        for token in self.special_tokens:
            encoded = token.encode("utf-8")
            if encoded not in self.byte_to_id:
                # 同步更新正向、反向词表；已经存在的标记保留原 ID。
                self.vocab[next_id] = encoded
                self.byte_to_id[encoded] = next_id
                next_id += 1
        # 单独建立 str -> ID，用于文本中遇到特殊标记时直接输出 ID。
        self.special_ids = {token: self.byte_to_id[token.encode("utf-8")]
                            for token in self.special_tokens}
        # 长标记优先；escape 让标记内部的 | 等符号按字面内容匹配。
        # join 中间添加的 | 则表示“匹配任意一种标记”。
        specials = "|".join(regex.escape(token) for token in
                            sorted(self.special_tokens, key=len, reverse=True))
        self.pattern = regex.compile(PAT)
        # A if condition else B 是条件表达式；没有标记就不建立特殊正则。
        self.special_pattern = regex.compile(specials) if specials else None
        # map(len, ...) 计算每个标记的字符长度，再取最大值；供流式缓冲使用。
        self.max_special_length = max(map(len, self.special_tokens), default=1)
        # 常见 pre-token 会重复出现；缓存有上限，不随文件大小无限增长。
        # lru_cache(...)(函数) 返回一个带缓存的新函数，接收相同输入时复用结果。
        # 前导下划线表示“内部辅助方法”，只是命名约定，不是访问权限限制。
        self._cached_encode_pretoken = lru_cache(maxsize=4096)(self._encode_pretoken)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                    special_tokens: list[str] | None = None) -> "Tokenizer":
        """读取实验脚本保存的两个 pickle 文件；仅加载自己信任的文件。"""
        # @classmethod 表示类方法：通过 Tokenizer.from_files(...) 调用，
        # 不需要先创建实例。cls 自动指向 Tokenizer 类，而不是某个实例。
        # 返回标注 "Tokenizer" 中加引号，是使用类名的前向引用写法。
        # with 在结束后自动关闭文件；rb 表示读取二进制文件。
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)
        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)
        return cls(vocab, merges, special_tokens)  # 相当于创建 Tokenizer(...)。

    def _encode_pretoken(self, text: str) -> tuple[int, ...]:
        # 只处理一个普通 pre-token；输入例如 ' cat'，输出是它的 ID 元组。
        # encode 返回 bytes；遍历 bytes 得到整数，bytes([value]) 转回单字节。
        # [...] 是列表推导式，相当于创建空列表后循环 append。
        tokens = [bytes([value]) for value in text.encode("utf-8")]
        while len(tokens) > 1:
            # 不比较频率：训练列表中位置越靠前，优先级越高。
            # zip(tokens, tokens[1:]) 枚举相邻 pair。
            # (...) 是生成器表达式：逐项提供候选，不先创建完整候选列表。
            # 只保留 merge_ranks 中已有的规则，绝不在编码时学习新规则。
            candidates = (pair for pair in zip(tokens, tokens[1:])
                            if pair in self.merge_ranks)
            # 字典.__getitem__(pair) 等价于 字典[pair]。
            # key 指定比较其 rank，而不是比较 pair 字典序。
            # 没有候选时返回 None；只有一个 token 时也不需要再合并。
            best_pair = min(candidates, key=self.merge_ranks.__getitem__, default=None)
            if best_pair is None:
                break
            updated = []
            i = 0
            # 与训练时一样，从左到右非重叠替换选中 pair。
            while i < len(tokens):
                # and 短路求值：先确保下一项存在，才访问 tokens[i+1]。
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair:
                    updated.append(tokens[i] + tokens[i + 1])
                    # 两个旧 token 被消耗，因此前进两位。
                    i += 2
                else:
                    updated.append(tokens[i])
                    i += 1
            tokens = updated
        # 把最终 bytes token 查成 ID；tuple(... for ...) 构造 ID 元组。
        # 缓存返回不可变元组，外部不容易意外修改缓存内容。
        return tuple(self.byte_to_id[token] for token in tokens)

    def _encode_piece(self, piece: str) -> Iterator[int]:
        # 含 yield 的函数是生成器函数：调用时返回迭代器，遍历时才执行。
        if piece in self.special_ids:
            # yield 产出一个 ID，并暂停；后续请求下一项时继续。
            yield self.special_ids[piece]
        else:
            # yield from 逐个产出另一个可迭代对象里的元素。
            # 这里产出普通 pre-token 对应的每个 ID，而不是整个元组。
            yield from self._cached_encode_pretoken(piece)

    def _matches(self, text: str):
        # 先隔离特殊标记，再匹配普通文本，防止普通正则吞掉嵌入词内的标记。
        start = 0
        if self.special_pattern:
            # finditer 逐个提供 Match 对象，而非匹配到的字符串。
            for special in self.special_pattern.finditer(text):
                # Match.start()/end() 是标记的起止字符下标，end 不包含在内。
                # 只预切分上一个标记之后、当前标记之前的普通文本。
                yield from self.pattern.finditer(text, start, special.start())
                # 特殊标记也作为一个完整匹配输出，确保保留原文顺序。
                yield special
                start = special.end()
        # 没有特殊标记时匹配全文；有标记时处理最后一个标记后的尾部。
        yield from self.pattern.finditer(text, start)

    def encode(self, text: str) -> list[int]:
        ids = []
        for match in self._matches(text):
            # group(0) 取完整匹配文本。
            # extend 将多个 ID 逐个加入列表；append 会把整个对象加入一项。
            ids.extend(self._encode_piece(match.group(0)))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """保留未确定的文本尾部，避免单词/缩写/特殊标记跨输入块被截断。

        不累积完整文件；缓存取决于未结束的 pre-token 和输入块大小。
        极长、无边界的单个 pre-token 仍可能需要较大缓冲。
        """
        pending = ""
        for chunk in iterable:
            # 输入块可能是一行，也可能只是一个字符，不能各自独立 encode。
            # 例如 ['hel','lo'] 必须按 'hello' 编码，而不是分别编码。
            pending += chunk
            matches = list(self._matches(pending))
            if len(matches) < 3:
                # 当前匹配太少，先等更多输入，不丢弃 pending。
                continue
            # 最后两个片段保留：追加字符可能改变尾部空白、缩写的匹配。
            # 另留最长特殊标记的长度，避免把尚未收齐的标记编码成普通文本。
            safe_limit = min(matches[-2].start(), len(pending) - self.max_special_length)
            # matches[-2] 是倒数第二个匹配；safe_limit 之前才考虑输出。
            # 边界是保守选择，不表示尾部文本无效，只是尚不确定能否定稿。
            consumed = 0
            for match in matches:
                if match.end() > safe_limit:
                    break
                yield from self._encode_piece(match.group(0))
                consumed = match.end()
            # 切掉已经编码的前缀，只留下未处理尾部；0 表示本轮没有输出。
            pending = pending[consumed:]
        # 输入耗尽，不会再有新字符，因此将最后缓冲全部编码输出。
        for match in self._matches(pending):
            yield from self._encode_piece(match.group(0))

    def decode(self, ids: list[int]) -> str:
        # 必须先拼接：单个 token 的 bytes 未必构成完整 UTF-8 字符。
        # 例如一个汉字的三个字节分在不同 token 中，单独解码会报错。
        # b''.join(...) 用空字节串连接全部 bytes，不添加任何分隔符。
        # errors='replace' 将无法解码的字节替换为 Unicode 替代字符 U+FFFD。
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")
