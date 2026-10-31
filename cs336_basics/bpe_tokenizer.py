import os
import heapq
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import regex
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def initialize(
    vocab: dict[int, bytes],
    special_tokens: list[str],
) -> None:
    for i in range(256):
        vocab[i] = bytes([i])

    for token_id, token in enumerate(special_tokens, start=256):
        vocab[token_id] = token.encode("utf-8")


def split_special_token(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> list[str]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not special_tokens:
        return [text]

    sorted_tokens = sorted(special_tokens, key=len, reverse=True) # key=len：按照每个字符串的长度比较。这里传入的是 len 函数，所以不加括号

    escaped_tokens = [] # 转译特殊字符
    for token in sorted_tokens:
        escaped_tokens.append(regex.escape(token))

    pattern = "|".join(escaped_tokens) # 写成正则
    parts = regex.split(pattern, text) # 用正则去切原文

    return parts

def pretokenization(
    parts: list[str]
) -> dict[str, int]:
    counts = Counter()
    compiled_pat = regex.compile(PAT)

    for part in parts:
        pretokens = compiled_pat.findall(part)
        counts.update(pretokens)

    return counts


def _count_file_chunk(task: tuple[str, int, int, str]) -> Counter[str]:
    """顶层 worker：只读取自己的字节范围，返回局部计数，不返回全文。"""
    path, start, end, special_token = task
    with open(path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8")
    # 分块边界在标记开头，标记仍归后一个块；这里将其排除。
    return Counter(pretokenization(text.split(special_token)))


def pretokenize_file_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_workers: int,
    chunk_bytes: int = 16 * 1024 * 1024,
) -> Counter[str]:
    """按文档边界分块计数。多标记/无标记情况保留原来的通用串行路径。

    使用多个较小任务，而不是每个 worker 读入整份文件的几分之一。
    边界搜索保留窗口重叠，避免漏掉跨 4096 字节窗口的特殊标记。
    """
    if num_workers < 1 or chunk_bytes < 1:
        raise ValueError("num_workers and chunk_bytes must be positive")
    if len(special_tokens) != 1 or not special_tokens[0] or num_workers == 1:
        return Counter(pretokenization(split_special_token(input_path, special_tokens)))

    path = os.fspath(input_path)
    marker = special_tokens[0].encode("utf-8")
    file_size = os.path.getsize(path)
    boundaries = [0]
    with open(path, "rb") as file:
        while boundaries[-1] + chunk_bytes < file_size:
            position = boundaries[-1] + chunk_bytes
            file.seek(position)
            while True:
                window = file.read(max(4096, len(marker)))
                found = window.find(marker)
                if found >= 0:
                    boundaries.append(position + found)
                    break
                if len(window) < max(4096, len(marker)):
                    break
                # 下一次从本窗口末尾前 len(marker)-1 字节开始。
                position += len(window) - len(marker) + 1
                file.seek(position)
            if boundaries[-1] < position:
                # 文件剩余部分没有标记，作为最后一个块处理。
                break
    if boundaries[-1] != file_size:
        boundaries.append(file_size)
    tasks = [(path, start, end, special_tokens[0])
            for start, end in zip(boundaries, boundaries[1:])]
    counts: Counter[str] = Counter()
    # spawn 不继承主进程的大对象；调用入口须有 __main__ 保护。
    with ProcessPoolExecutor(max_workers=num_workers,
                            mp_context=multiprocessing.get_context("spawn")) as pool:
        # 分批提交，避免已完成的局部 Counter 在主进程大量积压。
        for offset in range(0, len(tasks), num_workers):
            for partial_counts in pool.map(_count_file_chunk, tasks[offset:offset + num_workers]):
                counts.update(partial_counts)
    return counts

def encode_utf_8(
    counts: dict[str, int]
) -> dict[tuple[bytes, ...], int]:

    res: dict[tuple[bytes, ...], int] = {}
    
    for token, count in counts.items():
        bytestr = token.encode("utf-8")
        tokens = []
        for i in bytestr:
            tokens.append(bytes([i]))
        res[tuple(tokens)] = count

    return res

def select_the_pair(
    token_counts: dict[tuple[bytes, ...], int],    
) -> tuple[bytes, bytes] | None:
    pair_counts: dict[tuple[bytes, bytes], int] = {}

    for tokens, count in token_counts.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count

    if not pair_counts:
        return None

    return max(pair_counts, key=lambda pair: (pair_counts[pair], pair))

def train(
    merges: list[tuple[bytes, bytes]],
    vocab: dict[int, bytes],
    vocab_size: int,
    encoded_bytes: dict[tuple[bytes, ...], int]
) -> None:
    """
    首先是统计频率，然后看选中哪个pair
    选择 pair 时，频率相同还要按作业要求选择字典序更大的 pair。

    选中 (b"c", b"a") 后，这一轮需要同步完成三件事：
        - 将拼接结果 b"ca" 加入 vocab，分配新 ID；
        - 将 (b"c", b"a") 追加到 merges，记录学习顺序；
        - 在所有 pre-token 序列中替换这对元素

    循环有两个停止条件：
        1. len(vocab) 已达到目标 vocab_size；
        2. 即使词表还没满，也已经没有相邻 pair 可以合并。
    """
    while len(vocab) < vocab_size:
        best_pair = select_the_pair(encoded_bytes)
        if best_pair is None:
            break
            
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        new_counts: dict[tuple[bytes, ...], int] = {}

        for tokens, count in encoded_bytes.items():
            merged_tokens: list[bytes] = []
            i = 0

            while i < len(tokens):
                if(
                    i + 1 < len(tokens)
                    and (tokens[i], tokens[i + 1]) == best_pair
                ):
                    merged_tokens.append(new_token)
                    i+=2
                else:
                    merged_tokens.append(tokens[i])
                    i+=1

            merged_sequence = tuple(merged_tokens)
            new_counts[merged_sequence] = (
                new_counts.get(merged_sequence, 0) + count
            )

        encoded_bytes = new_counts

def train_indexed(
    merges: list[tuple[bytes, bytes]],
    vocab: dict[int, bytes],
    vocab_size: int,
    encoded_bytes: dict[tuple[bytes, ...], int],
) -> None:
    """增量维护 pair 频率，只重建包含选中 pair 的序列。

    序列 ID 始终不变；即使两个序列合并后相同，也各自保留频率。
    使用最大堆选择 pair，并惰性丢弃旧频率记录。

    与旧 train 的区别：旧版本每轮扫描全部序列并重算全部 pair；
    此版本先建立索引，之后只对包含 best_pair 的序列扣除旧贡献、
    执行替换、加入新贡献。每轮结束时，pair_counts 必须等于
    对当前全部序列进行全量统计的结果。

    vocab 和 merges 被原地修改，因此不需要返回值。
    encoded_bytes 只用于初始化，调用者传入的字典不会被修改。
    """
    # 四份长期维护的状态：
    # 1. sequences[id]：该 pre-token 当前的 token 元组，内容随 merge 改变。
    # list(字典) 得到键列表；字典的键与 values() 遵循相同的插入顺序，
    # 所以下面的 sequences[id] 与 frequencies[id] 一一对应。
    sequences = list(encoded_bytes)
    # 2. frequencies[id]：该 pre-token 在原始语料中的出现次数，始终不变。
    # 序列 ID 是这里的列表下标，不是 vocabulary 中的 token ID。
    frequencies = list(encoded_bytes.values())
    # 3. pair_counts[pair]：pair 在整个语料中的加权频率。 
    pair_counts: dict[tuple[bytes, bytes], int] = {} # 这里的这个计算pair count的作用是什么？
    # 4. pair_to_sequences[pair]：包含该 pair 的序列 ID 集合。
    # 集合只记录“在哪里”，不记录“出现几次”，重复 ID 会自动去重。
    pair_to_sequences: dict[tuple[bytes, bytes], set[int]] = {}

    def local_pairs(tokens: tuple[bytes, ...]) -> dict[tuple[bytes, bytes], int]:
        # 临时统计一个序列内部的 pair 次数，不乘该序列的语料频率。
        # 例如 (a,b,a,b) 得到 {(a,b): 2, (b,a): 1}。
        # counts 是局部临时字典，不是第五份长期维护的全局状态。
        counts: dict[tuple[bytes, bytes], int] = {}
        # tokens[1:] 去掉首元素；zip 将两边逐项配对：
        # (a,b,c) 与 (b,c) 配成 (a,b)、(b,c)。
        # 计数包括重叠位置：(a,a,a) 中 (a,a) 出现两次，
        # 但后续实际替换仍是不重叠的，这与朴素版保持一致。
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    # 初始化只全量扫描一次。enumerate 同时给出序列下标和内容。
    for sequence_id, tokens in enumerate(sequences):  #对每一个token开展计算，从而计算出paircount
        for pair, occurrences in local_pairs(tokens).items():
            # 局部出现 2 次、整个 pre-token 出现 3 次，贡献就是 2*3=6。
            pair_counts[pair] = pair_counts.get(pair, 0) + occurrences * frequencies[sequence_id]
            # setdefault：键不存在时创建空集合，存在时返回原集合；
            # add：将当前序列 ID 加入集合。
            pair_to_sequences.setdefault(pair, set()).add(sequence_id)

    class MaxPairEntry(tuple):
        """heapq 默认取最小项；反转比较，使最大 (频率, pair) 先出堆。"""

        def __lt__(self, other):
            # 连同 pair 字典序一起反转，保证同频时选字典序更大的 pair。
            return tuple.__gt__(self, other)

    # 堆中的频率是入堆时的快照；pair_counts 才是当前频率的权威来源。
    # 不能只给频率取负号，否则同频时会选到字典序较小的 pair。
    pair_heap = [MaxPairEntry((count, pair)) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap) # 会原地调整列表中的元素位置，建立堆结构；不需要接收返回值。

    # 两个继续条件：词表未满，并且仍有可合并 pair。
    while len(vocab) < vocab_size and pair_counts:
        # 频率更新时不在堆内部查找旧记录，而是追加新记录。
        # 弹出旧记录时，若与当前频率不一致或 pair 已删除，就跳过。
        while pair_heap:
            count, best_pair = heapq.heappop(pair_heap)
            if pair_counts.get(best_pair) == count:
                break
        else:
            raise RuntimeError("Pair heap is missing a live frequency record")
        # bytes 相加表示拼接。连续分配 ID 时，len(vocab) 是下一个 ID。
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        # 只处理索引命中的序列，其余序列和贡献原样保留。
        # 更新过程中集合会被修改甚至删除，因此先复制 ID，
        # 避免直接遍历正在变化的集合。
        affected_ids = tuple(pair_to_sequences[best_pair])
        # 这一轮结束后，只将受影响 pair 的最终频率推入堆，
        # 不为处理中间的每次扣减、增加都建立记录。
        changed_pairs: set[tuple[bytes, bytes]] = set()
        for sequence_id in affected_ids:
            tokens = sequences[sequence_id]
            frequency = frequencies[sequence_id]

            # 阶段 A：移除旧序列的全部贡献，其他序列的贡献不变。
            # 这里扣除的不仅是 best_pair，还包括会受邻接关系影响的 pair。
            # 例如 (a,b,c) 合并 (a,b)，旧 (b,c) 也必须消失。
            for pair, occurrences in local_pairs(tokens).items():
                changed_pairs.add(pair)
                pair_counts[pair] -= occurrences * frequency
                pair_to_sequences[pair].remove(sequence_id)
                # 无序列包含它时清理索引；频率归零时清理候选，
                # 已消失 pair 在堆中的旧记录将在弹出时被跳过。
                if not pair_to_sequences[pair]:
                    del pair_to_sequences[pair]
                if pair_counts[pair] == 0:
                    del pair_counts[pair]

            # 阶段 B：从左到右、不重叠地替换该序列。
            # tuple 不可原地修改，所以先构建列表，最后转回 tuple。
            merged_tokens: list[bytes] = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair:
                    # 匹配后消耗两个旧 token。例如 (a,a,a) -> (aa,a)，
                    # 中间的 a 不会重复参加第二次替换。
                    merged_tokens.append(new_token)
                    i += 2
                else:
                    # 没匹配则原样保留当前 token，只前进一位。
                    merged_tokens.append(tokens[i])
                    i += 1
            updated = tuple(merged_tokens)
            sequences[sequence_id] = updated

            # 阶段 C：加入新序列的全部贡献，并建立新的索引关联。
            # 上面先减旧贡献、这里再加新贡献，未变化的 pair 会抵消，
            # 因此无需手动判断哪些邻接位置改变了。
            # pre-token 的出现次数 frequency 不随合并改变。
            for pair, occurrences in local_pairs(updated).items():
                changed_pairs.add(pair)
                pair_counts[pair] = pair_counts.get(pair, 0) + occurrences * frequency
                pair_to_sequences.setdefault(pair, set()).add(sequence_id)

        # 所有受影响序列更新完，才发布最终频率并进行下一轮选择。
        for pair in changed_pairs:
            if pair in pair_counts:
                heapq.heappush(pair_heap, MaxPairEntry((pair_counts[pair], pair)))

        # 惰性删除会积累过期记录；超过阈值时重建，限制堆的额外内存。
        if len(pair_heap) > 4 * max(1, len(pair_counts)):
            pair_heap = [MaxPairEntry((count, pair)) for pair, count in pair_counts.items()]
            heapq.heapify(pair_heap)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int = 1,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    注释

    input
    1. 待训练的文件的路径
    2. 字典大小
    3. special token

    output
    1. 最终的字典，左边是int数字，右边是对应的byte
    2. 合并过程，merge list，用于encoding
    """

    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    initialize(vocab, special_tokens) # 初始化词表

    if num_workers == 1:
        parts = split_special_token(input_path, special_tokens)
        pre_tokens = pretokenization(parts)
        del parts  # merge 阶段不再保留全文片段。
    else:
        pre_tokens = pretokenize_file_parallel(input_path, special_tokens, num_workers)

    encoded_bytes = encode_utf_8(pre_tokens) # 然后将pretoken进行utf8编码 为什么还要拆？因为初始词表只有单字节 token；b"cat" 这个多字节 token 此时还没有通过 merge 学出来。
    del pre_tokens

    train_indexed(merges, vocab, vocab_size, encoded_bytes)

    return vocab, merges

