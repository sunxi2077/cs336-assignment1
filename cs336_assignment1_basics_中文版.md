# CS336 作业 1（基础）：构建一个 Transformer 语言模型

**版本 26.0.3**

CS336 课程组

2026 春季学期

## 1 作业概述

在本作业中，你需要从零开始构建训练一个标准 Transformer 语言模型（LM）所需的全部组件，并实际训练一些模型。

**你将实现的内容**

1. 字节对编码（BPE）分词器（第 2 节）
2. Transformer 语言模型（LM）（第 3 节）
3. 交叉熵损失函数与 AdamW 优化器（第 4 节）
4. 训练循环，支持模型与优化器状态的序列化保存和加载（第 5 节）

**你将运行的内容**

1. 在 TinyStories 数据集上训练 BPE 分词器。
2. 使用训练好的分词器对数据集进行编码，将其转换为整数 ID 序列。
3. 在 TinyStories 数据集上训练 Transformer LM。
4. 使用训练好的 Transformer LM 进行采样生成和困惑度（perplexity）评估。
5. 在 OpenWebText 上训练模型，并将你达到的困惑度提交到排行榜（leaderboard）。

**你可以使用的内容**

我们希望你从零开始构建每一个组件。特别地，除以下内容外，你**不得**使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义：

- `torch.nn.Parameter`
- `torch.nn` 中的容器类（例如 `Module`、`ModuleList`、`Sequential` 等）。¹
- `torch.optim.Optimizer` 基类

你可以使用任何其他 PyTorch 定义。如果你想使用某个函数或类但不确定是否允许，欢迎在 Slack 上提问。当不确定时，请考虑使用它是否会违背"从零开始"的作业精神。

**关于 AI 工具的声明**

AI 可以完全自主地完成作业中的许多部分，这使得深入接触和学习课程内容变得更加困难。

允许使用 AI 工具来回答高层次的概念性问题，或提供底层的编程文档，例如函数签名和库 API。但是，**不允许**使用 AI 工具来实现作业的任何部分。这既包括编程智能体（例如 Cursor Agents、Codex、Claude Code），也包括 AI 自动补全（例如 Cursor Tab、GitHub Copilot）。在使用 AI 智能体时，请确保它使用了提供的 `AGENTS.md` 文件。在使用聊天机器人时，也应当附上提示词（prompt）。

¹完整列表见 pytorch.org/docs/stable/nn.html#containers。

我们强烈建议你在完成作业时，在 IDE 中**禁用** AI 自动补全（例如 Cursor Tab、GitHub Copilot）（不过非 AI 的自动补全，例如自动补全函数名，完全没问题）。往届学生指出，禁用 AI 自动补全使他们更容易深入接触课程材料。

完整的 AI 政策见[此文档](https://github.com/stanford-cs336/assignment1-basics)。

**代码长什么样**

作业代码以及本文档均可在 GitHub 上获取：

github.com/stanford-cs336/assignment1-basics

请 `git clone` 该仓库。如果有任何更新，我们会通知你，以便你 `git pull` 获取最新版本。

1. `cs336_basics/*`：这是你编写代码的地方。注意这里没有任何代码——你可以完全从零开始按自己的方式实现！
2. `adapters.py`：有一组你的代码必须实现的功能。对于每一项功能（例如 scaled dot product attention），通过简单地调用你自己的代码来填写它的实现（例如 `run_scaled_dot_product_attention`）。注意：你对 `adapters.py` 的修改不应包含任何实质性的逻辑；这只是胶水代码。
3. `test_*.py`：包含你必须通过的所有测试（例如 `test_scaled_dot_product_attention`），它们会调用 `adapters.py` 中定义的钩子。不要编辑测试文件。

**如何提交**

为了提交，运行 `make_submission.sh` 来构建一个提交的 zip 文件。如果你有不想包含在提交 zip 中的大型数据文件或检查点（checkpoint），请务必把它们添加到脚本的排除列表中。

你需要向 Gradescope 提交以下文件：

- `writeup.pdf`：回答所有的书面问题。请对回答进行排版（typeset）。
- `code.zip`：包含你编写的所有代码。

要提交到排行榜，请向以下仓库提交 PR：

github.com/stanford-cs336/assignment1-basics-leaderboard

详细的提交说明见排行榜仓库中的 `README.md`。

**在哪里获取数据集**

本作业将使用两个预处理过的数据集：TinyStories [R. Eldan et al., 2023] 和 OpenWebText [A. Gokaslan et al., 2019]。这两个数据集都是单个的大型的纯文本文件。

如果你是随课程一起完成作业，可以在计算指南（compute guide）中找到下载数据集的说明。

如果你是在家自学，可以使用 `README.md` 中的命令下载这些文件。

> **低资源提示：引言（Init）**
>
> 在课程的各次作业讲义中，我们会给出一些建议，帮助你在 GPU 资源较少甚至没有 GPU 的情况下完成作业的各个部分。例如，我们有时会建议缩减数据集或模型规模，或者解释如何在 Mac 的集成 GPU 或 CPU 上运行训练代码。你会在蓝色框（像这个一样）中看到这些"低资源提示"。即使你是斯坦福的注册学生、可以使用课程机器，这些提示也可能帮助你更快地迭代并节省时间，所以我们建议你阅读它们！

> **低资源提示：在 Apple Silicon 或 CPU 上完成作业 1**
>
> 使用我们的参考实现（staff solution），我们可以在配备 36 GB 内存的 Apple M4 Max 芯片上训练一个能生成相当流畅文本的 LM：在 Metal GPU（MPS）上不到 5 分钟，使用 CPU 约 30 分钟。如果这些术语对你来说没什么意义，别担心！只需要知道：如果你有一台较新的笔记本电脑，并且你的实现正确且高效，你就能训练出一个能生成简单儿童故事、流畅度尚可的小型 LM。
>
> 在作业的后续部分，我们会说明如果你使用 CPU 或 MPS 需要做哪些调整。

## 2 字节对编码（BPE）分词器

在作业的第一部分，我们将训练并实现一个字节级（byte-level）的字节对编码（BPE）分词器 [R. Sennrich et al., 2016; C. Wang et al., 2019]。特别地，我们将把任意（Unicode）字符串表示为字节序列，并在这个字节序列上训练我们的 BPE 分词器。之后，我们将使用这个分词器把文本（字符串）编码成 token（整数序列），用于语言建模。

### 2.1 Unicode 标准

Unicode 是一种文本编码标准，它将字符映射到整数码点（code point）。截至 Unicode 17.0（2025 年 9 月发布），该标准在 172 种文字（script）中定义了 159,801 个字符。例如，字符 "s" 的码点是 115（通常记作 U+0073，其中 U+ 是惯例前缀，0073 是 115 的十六进制表示），字符 "牛" 的码点是 29275。在 Python 中，你可以使用 `ord()` 函数将单个 Unicode 字符转换为其整数表示。`chr()` 函数将一个整数 Unicode 码点转换为包含相应字符的字符串。

```
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

**问题（unicode1）：理解 Unicode（1 分）**

(a) `chr(0)` 返回什么 Unicode 字符？

**交付物：** 一句话回答。

(b) 该字符的字符串表示（`__repr__()`）与其打印（print）表示有何不同？

**交付物：** 一句话回答。

(c) 当这个字符出现在文本中时会发生什么？在你的 Python 解释器中尝试以下内容，看看它是否符合你的预期，这可能会有所帮助：

```
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

**交付物：** 一句话回答。

### 2.2 Unicode 编码

虽然 Unicode 标准定义了从字符到码点（整数）的映射，但直接在 Unicode 码点上训练分词器是不切实际的，因为词表会大到令人望而却步（约 150K 项）并且非常稀疏（因为许多字符相当罕见）。相反，我们将使用一种 Unicode **编码**（encoding），它把一个 Unicode 字符转换成一个字节序列。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上的主流编码（超过 98% 的网页使用它）。

要将 Unicode 字符串编码为 UTF-8，我们可以使用 Python 中的 `encode()` 函数。要访问 Python `bytes` 对象的底层字节值，我们可以迭代它（例如调用 `list()`）。最后，我们可以使用 `decode()` 函数将 UTF-8 字节串解码为 Unicode 字符串。

```
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # 获取编码后字符串的字节值（0 到 255 之间的整数）。
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129,
161, 227, 129, 175, 33]
>>> # 一个字节并不一定对应一个 Unicode 字符！
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

通过把我们的 Unicode 码点转换成字节序列（例如通过 UTF-8 编码），我们本质上是把一个码点序列（21 位整数，有 159,801 个有效值）转换成一个字节值序列（0 到 255 范围内的整数）。长度为 256 的字节词表要容易处理得多。使用字节级分词时，我们不需要担心词表外（out-of-vocabulary）token，因为我们知道任何输入文本都可以表示为 0 到 255 之间的整数序列。

**问题（unicode2）：Unicode 编码（3 分）**

(a) 有哪些理由让我们更倾向于在 UTF-8 编码的字节上训练分词器，而不是 UTF-16 或 UTF-32？比较这些编码对各种输入字符串的输出可能会有所帮助。

**交付物：** 一到两句话的回答。

(b) 考虑以下（不正确的）函数，它本意是将 UTF-8 字节串解码为 Unicode 字符串。为什么这个函数是不正确的？请给出一个会产生错误结果的输入字节串示例。

```
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

**交付物：** 一个使 `decode_utf8_bytes_to_str_wrong` 产生错误输出的输入字节串示例，并用一句话解释该函数为何不正确。

(c) 给出一个无法解码为任何 Unicode 字符的两字节序列。

**交付物：** 一个示例，并用一句话解释。

### 2.3 子词分词

虽然字节级分词可以缓解词级分词器面临的词表外问题，但把文本分词成字节会导致极长的输入序列。这会减慢模型训练，因为一个有 10 个单词的句子在词级语言模型中可能只有 10 个 token，但在字符级模型中可能长达 50 个或更多 token（取决于单词的长度）。处理这些更长的序列需要在模型的每一步进行更多计算。此外，在字节序列上进行语言建模是困难的，因为更长的输入序列会在数据中产生长距离依赖。

子词分词是词级分词器和字节级分词器之间的折中。注意，字节级分词器的词表有 256 个条目（字节值为 0 到 255）。子词分词器用更大的词表来换取对输入字节序列更好的压缩。例如，如果字节序列 `b'the'` 在我们的原始文本训练数据中经常出现，为它在词表中分配一个条目，就可以把这个 3-token 的序列缩减为单个 token。

我们如何选择要加入词表的这些子词单元？R. Sennrich et al. [3] 提出使用字节对编码（BPE；P. Gage [5]），这是一种压缩算法，它迭代地将最频繁出现的字节对替换（"合并"）为一个新的、未使用的索引。注意，该算法向词表添加子词 token 是为了最大化对输入序列的压缩——如果某个单词在输入文本中出现的次数足够多，它就会被表示为一个单独的子词单元。

通过 BPE 构造词表的子词分词器通常被称为 BPE 分词器。在本作业中，我们将实现一个字节级 BPE 分词器，其中词表条目是字节或字节的合并序列，这样我们就能在词表外处理和可管理的输入序列长度两方面兼得。构造 BPE 分词器词表的过程被称为"训练"BPE 分词器。

### 2.4 BPE 分词器训练

BPE 分词器的训练过程包括三个主要步骤。

**词表初始化**

分词器词表是从字节串 token 到整数 ID 的一一映射。由于我们训练的是字节级 BPE 分词器，我们的初始词表就是所有字节的集合。由于有 256 个可能的字节值，我们的初始词表大小为 256。

**预分词（Pre-tokenization）**

一旦有了词表，原则上你可以统计字节在文本中彼此相邻出现的频率，并从最频繁的字节对开始合并。然而，这在计算上相当昂贵，因为每次合并我们都必须完整遍历一遍语料库。此外，直接在语料库上合并字节可能会导致生成的 token 只在标点符号上不同（例如 `dog!` 和 `dog.`）。这些 token 会得到完全不同的 token ID，尽管它们很可能具有很高的语义相似度（因为它们只在标点符号上不同）。

为了避免这一点，我们对语料库进行**预分词**。你可以把它看作是对语料库的一种粗粒度分词，它帮助我们统计字符对出现的频率。例如，单词 `'text'` 可能是一个出现了 10 次的预分词（pre-token）。在这种情况下，当我们统计字符 't' 和 'e' 相邻出现的频率时，我们会看到单词 'text' 中 't' 和 'e' 是相邻的，于是可以把它们的计数加 10，而无需遍历语料库。由于我们训练的是字节级 BPE 模型，每个预分词都表示为一个 UTF-8 字节序列。

R. Sennrich et al. [3] 的原始 BPE 实现通过简单地按空白符切分来进行预分词（即 `s.split(" ")`）。这种方法在基于 SentencePiece 的分词器（例如 Llama 1 和 2 的分词器）中仍然存在。

大多数现代分词器使用基于正则表达式的预分词器，这是 GPT-2 的做法；A. Radford et al. [6]。我们将使用原始正则表达式的一个略微美化版本，取自 github.com/openai/tiktoken/pull/234/files：

```
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

交互式地用这个预分词器切分一些文本，以更好地理解它的行为，可能会很有用：

```
>>> # 需要 `regex` 包
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

不过，在你的代码中使用它时，应该使用 `re.finditer`，以避免在构建"预分词到计数"的映射时存储已预分词的单词。

**计算 BPE 合并**

现在我们已经把输入文本转换成了预分词，并把每个预分词表示为一个 UTF-8 字节序列，我们就可以计算 BPE 合并（即训练 BPE 分词器）了。在高层次上，BPE 算法迭代地统计每一个字节对，并找出频率最高的字节对（"A", "B"）。然后，这个最频繁字节对（"A", "B"）的每次出现都被合并，即替换为一个新 token "AB"。这个新的合并 token 被加入我们的词表；因此，BPE 训练后的最终词表大小等于初始词表大小（在我们的例子中是 256）加上训练期间执行的 BPE 合并操作的次数。为了在 BPE 训练期间提高效率，我们不考虑跨越预分词边界的字节对。² 在计算合并时，通过**优先选择字典序更大的字节对**来确定性打破平局。例如，如果字节对 ("A", "B")、("A", "C")、("B", "ZZ") 和 ("BA", "A") 都具有最高频率，我们会合并 ("BA", "A")：

```
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

²注意，原始 BPE 公式 R. Sennrich et al. [3] 规定了要加入一个词尾（end-of-word）token。在训练字节级 BPE 模型时，我们不添加词尾 token，因为所有字节（包括空白符和标点符号）都已包含在模型的词表中。由于我们显式地表示空格和标点符号，学习到的 BPE 合并会自然地反映这些词边界。

**特殊 token**

通常，一些字符串（例如 `<|endoftext|>`）被用来编码元数据（例如文档之间的边界）。在编码文本时，通常希望把某些字符串视为"特殊 token"，它们永远不应被拆分成多个 token（即总是被保留为单个 token）。例如，序列结束字符串 `<|endoftext|>` 应始终被保留为单个 token（即单个整数 ID），这样我们就知道何时停止从语言模型生成。这些特殊 token 必须被加入词表，以便它们有对应的固定 token ID。

R. Sennrich et al. [3] 的算法 1 包含一个低效的 BPE 分词器训练实现（本质上遵循我们上面概述的步骤）。作为第一个练习，实现并测试这个函数以检验你的理解，可能会很有用。

**示例（bpe_example）：BPE 训练示例**

下面是 R. Sennrich et al. [3] 中的一个程式化示例。考虑由以下文本组成的语料库

```
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

并且词表有一个特殊 token `<|endoftext|>`。

**词表**

我们用特殊 token `<|endoftext|>` 和 256 个字节值初始化词表。

**预分词**

为了简单并聚焦于合并过程，我们在这个示例中假设预分词只是按空白符切分。当我们预分词并计数后，得到如下频率表。

```
{low: 5, lower: 2, widest: 3, newest: 6}
```

将其表示为 `dict[tuple[bytes, ...], int]` 比较方便，例如 `{(l,o,w): 5, …}`。注意，在 Python 中即使单个字节也是一个 `bytes` 对象。Python 中没有表示单个字节的 byte 类型，就像 Python 中没有表示单个字符的 char 类型一样。

**合并**

我们首先查看每一对连续字节，并对其出现的单词频率求和 `{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}`。字节对 ('e', 's') 和 ('s', 't') 并列，所以我们取字典序更大的字节对 ('s', 't')。然后我们合并预分词，得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}`。

在第二轮中，我们看到 (e, st) 是最常见的字节对（计数为 9），于是合并得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}`。继续这个过程，我们最终得到的合并序列将是 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。

如果我们取 6 次合并，则得到 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，我们的词表元素将是 `[<|endoftext|>, [...256 个字节字符], st, est, ow, low, west, ne]`。

有了这个词表和这组合并，单词 `newest` 将被分词为 `[ne, west]`。

### 2.5 BPE 分词器训练实验

让我们在 TinyStories 数据集上训练一个字节级 BPE 分词器。查找/下载数据集的说明见第 1 节。开始之前，我们建议先看一下 TinyStories 数据集，感受一下数据中有什么。

**并行化预分词**

你会发现预分词步骤是一个主要瓶颈。你可以通过使用内置的 `multiprocessing` 库并行化你的代码来加速预分词。具体来说，我们建议在预分词的并行实现中，对语料库进行分块（chunk），同时确保块边界出现在特殊 token 的开头。你可以直接逐字使用以下链接中的起始代码来获取块边界，然后用它在各进程之间分配工作：

https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py

这种分块方式总是有效的，因为我们绝不想跨越文档边界进行合并。就本作业而言，你总是可以这样切分。不必担心遇到一个不含 `<|endoftext|>` 的超大语料库这种边缘情况。

**预分词前移除特殊 token**

在使用正则表达式模式（通过 `re.finditer`）运行预分词之前，你应该从语料库（或使用并行实现时的块）中剥离所有特殊 token。请确保按特殊 token 进行切分，这样就不会在它们所分隔的文本之间发生任何合并。例如，如果你有一个语料库（或块），形如 `[Doc 1]<|endoftext|>[Doc 2]`，你应该按特殊 token `<|endoftext|>` 切分，并分别预分词 `[Doc 1]` 和 `[Doc 2]`，这样就不会跨越文档边界发生合并。换句话说，特殊 token 在训练期间定义了硬分割边界，但它们本身不应贡献合并计数。这可以通过 `re.split` 来实现，使用 `"|".join(special_tokens)` 作为分隔符（由于特殊 token 中可能出现 `|`，需谨慎使用 `re.escape`）。测试 `test_train_bpe_special_tokens` 将对此进行测试。

**优化合并步骤**

上面程式化示例中 BPE 训练的朴素实现很慢，因为每次合并都要遍历所有字节对来找出最频繁的字节对。然而，每次合并后发生变化的字节对计数只有那些与合并字节对重叠的字节对。因此，可以通过为所有字节对的计数建立索引并增量更新这些计数，来提高 BPE 训练速度，而不是显式地遍历每一对字节来统计字节对频率。通过这种缓存过程你可以获得显著的加速，不过我们注意到 BPE 训练的合并部分在 Python 中是无法并行化的。

> **低资源提示：性能分析（Profiling）**
>
> 你应该使用 `cProfile` 或 `py-spy` 等性能分析工具来找出实现中的瓶颈，并专注于优化这些瓶颈。

> **低资源提示："缩减规模"（Downscaling）**
>
> 与其直接跳转到在完整的 TinyStories 数据集上训练分词器，我们建议你先在数据的一个小子集上训练：一个"调试数据集"。例如，你可以先在 TinyStories 验证集上训练你的分词器，验证集是 22K 篇文档，而不是 2.12M 篇。这说明了只要可能就缩减规模以加快开发速度的通用策略：例如使用更小的数据集、更小的模型规模等。选择调试数据集或超参数配置的大小需要仔细考虑：你希望调试集足够大，以具备与完整配置相同的瓶颈（这样你做的优化才能泛化），但又不能大到运行起来没完没了。

**问题（train_bpe）：BPE 分词器训练（15 分）**

**交付物：** 编写一个函数，给定输入文本文件的路径，训练一个（字节级）BPE 分词器。你的 BPE 训练函数应至少处理以下输入参数：

**输入**

- `input_path: str` —— 包含 BPE 分词器训练数据的文本文件的路径。
- `vocab_size: int` —— 一个正整数，定义最终词表的最大大小（包括初始字节词表、由合并产生的词表条目以及任何特殊 token）。
- `special_tokens: list[str]` —— 要加入词表的字符串列表。在训练期间，将它们视为硬边界，阻止跨越其范围的合并，但在计算合并统计时不包括它们。

你的 BPE 训练函数应返回最终的词表和合并：

**输出**

- `vocab: dict[int, bytes]` —— 分词器词表，一个从 `int`（词表中的 token ID）到 `bytes`（token 字节）的映射。
- `merges: list[tuple[bytes, bytes]]` —— 训练产生的 BPE 合并列表。每个列表项是一个 `bytes` 元组 `(<token1>, <token2>)`，表示 `<token1>` 与 `<token2>` 合并。合并应按创建顺序排列。

要针对我们提供的测试来测试你的 BPE 训练函数，你首先需要实现位于 `[adapters.run_train_bpe]` 的测试适配器。然后运行 `uv run pytest tests/test_train_bpe.py`。你的实现应能通过所有测试。（可选，这可能是一项较大的时间投入）你可以使用某种系统语言实现训练方法的关键部分，例如 C++（考虑 cppyy 或 nanobind）或 Rust（使用 PyO3）。如果你这样做，请注意哪些操作需要复制数据，哪些直接从 Python 内存中读取，并确保留下构建说明，或确保它仅使用 `pyproject.toml` 就能构建。还要注意，GPT-2 的正则表达式在大多数正则引擎中得不到很好的支持，在多数支持它的引擎中也会太慢。我们已验证 Oniguruma 速度相当快且支持负向前瞻（negative lookahead），但 Python 中的 `regex` 包即使不比它快，也至少同样快。

**问题（train_bpe_tinystories）：在 TinyStories 上训练 BPE（2 分）**

(a) 在 TinyStories 数据集上训练一个字节级 BPE 分词器，最大词表大小为 10,000。请务必将 TinyStories 的 `<|endoftext|>` 特殊 token 加入词表。将得到的词表和合并序列化到磁盘以便进一步检查。训练花了多少时间和内存？词表中最长的 token 是什么？它合理吗？

**资源要求：** ≤ 30 分钟（无 GPU），≤ 30 GB 内存

**提示** —— 通过在预分词期间使用多进程，并利用以下两个事实，你应该能在 2 分钟内完成 BPE 训练：

(a) `<|endoftext|>` token 在数据文件中分隔文档。

(b) 在应用 BPE 合并之前，将 `<|endoftext|>` token 作为特殊情况处理。

**交付物：** 一到两句话的回答。

(b) 对你的代码进行性能分析。分词器训练过程中哪个部分花费的时间最多？

**交付物：** 一到两句话的回答。

接下来，我们尝试在 OpenWebText 数据集上训练字节级 BPE 分词器。和之前一样，我们建议先看一下数据集以更好地理解其内容。

**问题（train_bpe_expts_owt）：在 OpenWebText 上训练 BPE（2 分）**

(a) 在 OpenWebText 数据集上训练一个字节级 BPE 分词器，最大词表大小为 32,000。将得到的词表和合并序列化到磁盘以便进一步检查。词表中最长的 token 是什么？它合理吗？

**资源要求：** ≤ 12 小时（无 GPU），≤ 100 GB 内存

**交付物：** 一到两句话的回答。

(b) 比较在 TinyStories 与 OpenWebText 上训练得到的分词器。

**交付物：** 一到两句话的回答。

### 2.6 BPE 分词器：编码与解码

在作业的前一部分，我们实现了一个函数，在输入文本上训练 BPE 分词器以获得分词器词表和 BPE 合并列表。现在，我们将实现一个 BPE 分词器，它加载给定的词表和合并列表，并用它们在文本与 token ID 之间进行编码和解码。

#### 2.6.1 编码文本

BPE 编码文本的过程与我们训练 BPE 词表的方式相对应。有几个主要步骤。

**第 1 步：预分词。** 我们首先对序列进行预分词，并把每个预分词表示为一个 UTF-8 字节序列，就像 BPE 训练时那样。我们将在每个预分词内部把这些字节合并成词表元素，独立处理每个预分词（不跨越预分词边界进行合并）。

**第 2 步：应用合并。** 然后，我们取 BPE 训练期间创建的词表元素合并序列，并按创建顺序将它应用到我们的预分词上。

**示例（bpe_encoding）：BPE 编码示例**

例如，假设我们的输入字符串是 `'the cat ate'`，我们的词表是 `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}`，我们学习到的合并是 `[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]`。首先，我们的预分词器会把该字符串切分为 `['the', ' cat', ' ate']`。然后，我们查看每个预分词并应用 BPE 合并。

第一个预分词 `'the'` 初始表示为 `[b't', b'h', b'e']`。查看我们的合并列表，我们识别出第一个可应用的合并是 `(b't', b'h')`，用它把预分词转换为 `[b'th', b'e']`。然后，我们回到合并列表，识别出下一个可应用的合并是 `(b'th', b'e')`，它把预分词转换为 `[b'the']`。最后，回头看合并列表，我们发现没有更多适用于该字符串的合并了（因为整个预分词已被合并成单个 token），于是完成了 BPE 合并的应用。对应的整数序列是 `[9]`。

对其余预分词重复这一过程，我们看到预分词 `' cat'` 在应用 BPE 合并后表示为 `[b' c', b'a', b't']`，变成整数序列 `[7, 1, 5]`。最后一个预分词 `' ate'` 在应用 BPE 合并后为 `[b' at', b'e']`，变成整数序列 `[10, 3]`。因此，编码输入字符串的最终结果是 `[9, 7, 1, 5, 10, 3]`。

**特殊 token**

你的分词器在编码文本时应能正确处理用户定义的特殊 token（在构造分词器时提供）。

**内存考虑**

假设我们想分词一个无法放入内存的大型文本文件。为了高效地分词这个大文件（或任何其他数据流），我们需要把它拆分成可管理的块，并依次处理每个块，这样内存复杂度相对于文本大小就是常数而非线性。在这样做时，我们需要确保不会有 token 跨越块边界，否则我们会得到与在内存中分词整个序列的朴素方法不同的分词结果。

#### 2.6.2 解码文本

要把整数 token ID 序列解码回原始文本，我们可以简单地查每个 ID 在词表中对应的条目（一个字节序列），把它们拼接起来，然后把字节解码为 Unicode 字符串。注意，输入的 ID 并不保证映射到有效的 Unicode 字符串（因为用户可以输入任意的整数 ID 序列）。当输入 token ID 不产生有效 Unicode 字符串时，你应该用官方的 Unicode 替换字符 U+FFFD 替换有问题的字节。³ `bytes.decode` 的 `errors` 参数控制如何处理 Unicode 解码错误，使用 `errors='replace'` 会自动用替换标记替换有问题的数据。

³关于 Unicode 替换字符的更多信息见 en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character。

**问题（tokenizer）：实现分词器（15 分）**

**交付物：** 实现一个 `Tokenizer` 类，给定一个词表和一个合并列表，将文本编码为整数 ID，并将整数 ID 解码为文本。你的分词器还应支持用户提供的特殊 token（如果它们尚不在词表中，则将它们追加到词表中）。我们推荐以下接口：

```
def __init__(self, vocab, merges, special_tokens=None)
```

从给定的词表、合并列表和（可选的）特殊 token 列表构造一个分词器。此函数应接受以下参数：

- `vocab: dict[int, bytes]`
- `merges: list[tuple[bytes, bytes]]`
- `special_tokens: list[str] | None = None`

```
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
```

类方法，从一个序列化的词表和合并列表（格式与你的 BPE 训练代码输出相同）以及（可选的）特殊 token 列表构造并返回一个 `Tokenizer`。此方法应接受以下附加参数：

- `vocab_filepath: str`
- `merges_filepath: str`
- `special_tokens: list[str] | None = None`

```
def encode(self, text: str) -> list[int]
```

将输入文本编码为 token ID 序列。

```
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
```

给定一个字符串的可迭代对象（例如 Python 文件句柄），返回一个惰性产出 token ID 的生成器。这是对无法直接加载到内存的大型文件进行内存高效分词所必需的。

```
def decode(self, ids: list[int]) -> str
```

将 token ID 序列解码为文本。

要针对我们提供的测试来测试你的 `Tokenizer`，你首先需要实现位于 `[adapters.get_tokenizer]` 的测试适配器。然后运行 `uv run pytest tests/test_tokenizer.py`。你的实现应能通过所有测试。

### 2.7 实验

**问题（tokenizer_experiments）：分词器实验（4 分）**

(a) 从 TinyStories 和 OpenWebText 中抽样 10 篇文档。使用你之前训练好的 TinyStories 和 OpenWebText 分词器（词表大小分别为 10K 和 32K），把这些抽样文档编码为整数 ID。每个分词器的压缩比（bytes/token）是多少？

**交付物：** 一到两句话的回答。

(b) 如果你用 TinyStories 分词器来分词 OpenWebText 样本，会发生什么？比较压缩比和/或定性描述发生的情况。

**交付物：** 一到两句话的回答。

(c) 估算你分词器的吞吐量（例如以 bytes/second 为单位）。分词 Pile 数据集（825GB 文本）需要多长时间？

**交付物：** 一到两句话的回答。

(d) 使用你的 TinyStories 和 OpenWebText 分词器，把相应的训练集和开发集编码为整数 token ID 序列。我们稍后将用它来训练语言模型。我们建议把 token ID 序列化为 `uint16` 数据类型的 NumPy 数组。为什么 `uint16` 是合适的选择？

**交付物：** 一到两句话的回答。

## 3 Transformer 语言模型架构

语言模型接收一个批量的整数 token ID 序列（即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`），并返回一个（批量的）在词表上的归一化概率分布（即形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch Tensor），其中预测的分布是对每个输入 token 的**下一个词**的预测。在训练语言模型时，我们使用这些下一个词预测来计算实际下一个词与预测下一个词之间的交叉熵损失。在推理时从语言模型生成文本时，我们取最后一个时间步（即序列中的最后一项）的预测下一个词分布来生成序列中的下一个 token（例如取概率最高的 token、从分布中采样等），把生成的 token 加入输入序列，然后重复这一过程。

在本作业的这一部分，你将从头构建这个 Transformer 语言模型。我们将先对模型进行高层次描述，然后逐步详细介绍各个组件。

### 3.1 Transformer LM

给定一个 token ID 序列，Transformer 语言模型使用输入嵌入（input embedding）把 token ID 转换为稠密向量，把嵌入后的 token 通过 `num_layers` 个 Transformer 块，然后应用一个可学习的线性投影（"输出嵌入"或"LM head"）来产生预测的下一个 token 的 logits。示意表示见图 1。

> **图 1**：我们的 Transformer 语言模型概览。
>
> [Token Embedding（Token 嵌入） → Inputs（输入）→ num_layers 个 Transformer Block → Norm → Linear (Output Embedding)（输出嵌入）→ Softmax → Output Probabilities（输出概率）]
>
> **图 2**：一个 pre-norm Transformer 块。
>
> 输入张量形状为 `(batch_size, seq_len, d_model)`；块内依次为：Norm → Causal Multi-Head Self-Attention w/ RoPE（带 RoPE 的因果多头自注意力）→ Add（残差相加）；Norm → Position-Wise Feed-Forward（逐位置前馈网络）→ Add（残差相加）；输出张量形状为 `(batch_size, seq_len, d_model)`。

**Token 嵌入**

在最初一步，Transformer 把（批量的）token ID 序列嵌入为包含 token 身份信息的向量序列（图 1 中的红色块）。

更具体地说，给定一个 token ID 序列，Transformer 语言模型使用一个 token 嵌入层来产生一个向量序列。每个嵌入层接收形状为 `(batch_size, sequence_length)` 的整数张量，并产生形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

**Pre-norm Transformer 块**

嵌入之后，激活值由若干个结构相同的神经网络层处理。一个标准的 decoder-only Transformer 语言模型由 `num_layers` 个相同的层（通常称为 Transformer "块"）组成。每个 Transformer 块接收形状为 `(batch_size, sequence_length, d_model)` 的输入，并返回形状为 `(batch_size, sequence_length, d_model)` 的输出。每个块在序列上聚合信息（通过自注意力）并对它进行非线性变换（通过前馈层）。

经过 `num_layers` 个 Transformer 块后，我们取最终的激活值，把它变成在词表上的分布。

我们将实现"pre-norm"Transformer 块（详见 3.4 节），它另外要求在最后一个 Transformer 块之后使用层归一化（详见下文），以确保其输出被适当地缩放。

在此归一化之后，我们将使用一个标准的可学习线性变换，把 Transformer 块的输出转换为预测的下一个 token 的 logits（例如见 A. Radford et al. [7] 公式 2）。

### 3.2 说明：批处理、Einsum 与高效计算

在整个 Transformer 中，我们将对许多类批量的输入执行相同的计算。这里有几个例子：

- **批中的元素**：我们对每个批量元素应用相同的 Transformer 前向操作。
- **序列长度**："逐位置"操作（如 RMSNorm 和前馈网络）对序列的每个位置做完全相同的操作。
- **注意力头**：在"多头"注意力操作中，注意力操作在多个注意力头上进行批处理。

拥有一种能充分利用 GPU、又易于阅读和理解的、符合人体工学的操作方式会很有用。许多 PyTorch 操作可以接收张量开头的额外"类批量"维度，并在这些维度上高效地重复/广播操作。

例如，假设我们正在做一个逐位置、批量的操作。我们有一个形状为 `(batch_size, sequence_length, d_model)` 的"数据张量" 𝐷，想对它和一个形状为 `(d_model, d_model)` 的矩阵 𝐴 做批量向量-矩阵乘法。在这种情况下，`D @ A` 会做一个批量矩阵乘法，这是 PyTorch 中一个高效的原语，其中 `(batch_size, sequence_length)` 维度被批量处理。

因此，假设你的函数可能会收到额外的类批量维度，并把那些维度放在 PyTorch 形状的开头，会很有帮助。为了以这种方式组织张量以便批处理，它们可能需要通过多步 `view`、`reshape` 和 `transpose` 来整形。这可能有点麻烦，而且往往很难看出代码在做什么以及张量的形状是什么。

一种更符合人体工学的选择是使用 `torch.einsum` 中的 einsum 表示法，或者使用与框架无关的库如 `einops` 或 `einx`。两个关键操作是 `einsum`（可以对输入张量的任意维度做张量缩并）和 `rearrange`（可以重排、拼接和拆分任意维度）。事实证明，机器学习中几乎所有的操作都是某种维度操作和张量缩并的组合，偶尔加上一个（通常是逐点的）非线性函数。这意味着使用 einsum 表示法时，你的很多代码可以更可读、更灵活。

我们强烈建议在本课程中学习并使用 einsum 表示法。尚未接触过 einsum 表示法的同学应使用 einops（文档见[这里](https://einops.rocks/)），而已经熟悉 einops 的同学应学习更通用的 einx（见[这里](https://einx.readthedocs.io/)）。⁴ 这两个包都已经安装在所提供的环境中。

这里我们给出一些 einsum 表示法用法的例子。这些是对 einops 文档的补充，你应该先阅读 einops 文档。

⁴值得注意的是，虽然 einops 有很好的支持，但 einx 还没有那么久经考验。如果你发现 einx 有任何限制或 bug，请随时回退到使用 einops 并辅以一些普通 PyTorch。

**示例（einstein_example1）：用 einops.einsum 做批量矩阵乘法**

```
import torch
from einops import rearrange, einsum
## 基本实现
Y = D @ A.T
# 很难看出输入和输出形状以及它们的含义。
# D 和 A 可以有什么形状，其中哪些会有意外的行为？
## Einsum 是自文档化的且健壮
#                          D                A     ->          Y
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
## 或者，一个批量版本，D 可以有任意前导维度，但 A 受约束。
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

**示例（einstein_example2）：用 einops.rearrange 做广播操作**

我们有一批图像，对于每张图像，我们想根据某个缩放因子生成 10 个变暗版本：

```
images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)
## 重塑并相乘
dim_value = rearrange(dim_by,    "dim_value              -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value
## 或者一步完成：
dimmed_images = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)
```

**示例（einstein_example3）：用 einops.rearrange 做像素混合**

假设我们有一批图像，表示为形状为 `(batch, height, width, channel)` 的张量，我们想对图像的所有像素做线性变换，但该变换应独立地作用于每个通道。我们的线性变换表示为形状为 `(height * width, height * width)` 的矩阵 𝐵。

```
channels_last = torch.randn(64, 32, 32, 3)  # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)
## 重塑图像张量以便跨所有像素混合
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
```

改用 einops：

```
height = width = 32
## Rearrange 取代了笨拙的 torch view + transpose
channels_first = rearrange(
    channels_last,
    "batch height width channel -> batch channel (height width)"
)
channels_first_transformed = einsum(
    channels_first, B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height=height, width=width
)
```

或者，如果你胆大：用 einx.dot（einx 中等价于 einops.einsum 的操作）一步完成

```
height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last, B,
    col_in=width, col_out=width
)
```

这里的第一个实现可以通过在前后添加注释来标明输入和输出形状，但这很笨拙且容易出 bug。使用 einsum 表示法，文档就是实现！

Einsum 表示法可以处理任意的输入批量维度，而且还有一个关键好处是自文档化。在使用 einsum 表示法的代码中，输入和输出张量的相关形状要清楚得多。对于其余张量，你可以考虑使用 Tensor 类型提示，例如使用 `jaxtyping` 库（并非 JAX 专用）。

关于使用 einsum 表示法的性能影响，我们将在作业 2 中进一步讨论，但现在你只需要知道：它们几乎总是比替代方案更好！

#### 3.2.1 数学记号与内存顺序

许多机器学习论文在记号中使用行向量，这会产生与 NumPy 和 PyTorch 默认使用的行主序（row-major）内存顺序很好契合的表示。使用行向量时，线性变换看起来像

𝑦 = 𝑥𝑊⊤,    (1)

其中行主序 𝑊 ∈ ℝ^{d_out×d_in}，行向量 𝑥 ∈ ℝ^{1×d_in}。注意，这让我们可以通过增加 𝑥 的最外层维度来批处理输入，也就是说，我们可以把向量输入 𝑥 替换为矩阵输入 𝑋 ∈ ℝ^{batch×d_in}。

在线性代数中，通常更常用列向量，此时线性变换看起来像

𝑦 = 𝑊𝑥,    (2)

给定行主序的 𝑊 ∈ ℝ^{d_out×d_in} 和列向量 𝑥 ∈ ℝ^{d_in}。在这种情况下要批处理输入，𝑥 的批量维度必须放在最后，所以 𝑥 需要被替换为矩阵 𝑋̃ ∈ ℝ^{d_in×batch}。

在本作业中，我们在数学记号中主要使用列向量，因为数学通常遵循这种记号。你应该记住：如果你想使用普通矩阵乘法记号，你必须像公式 1 的行向量约定那样，对矩阵施加转置，因为 PyTorch 使用行主序内存顺序。如果你用 einsum 来做线性代数运算，只要你正确地标注了轴，这就不是问题。顺便说一句，值得注意的是，其他语言/线性代数包（如 Matlab、Julia 和 Fortran）都使用列主序内存顺序，意味着批量维度放在最后，但 Python 及相关包采用了 C 标准的行主序顺序。

### 3.3 基本构建块：Linear 与 Embedding 模块

#### 3.3.1 参数初始化

有效训练神经网络通常需要仔细初始化模型参数——糟糕的初始化会导致梯度消失或梯度爆炸等不良行为。Pre-norm Transformer 对初始化异常鲁棒，但初始化仍会对训练速度和收敛产生显著影响。由于本作业已经很长，我们把细节留到作业 3，这里先给你一些在大多数情况下应该有效的近似初始化。目前请使用：

- **Linear 权重**：𝒩(𝜇 = 0, 𝜎² = 2/(d_in+d_out))，截断到 [−3𝜎, 3𝜎]。
- **Embedding**：𝒩(𝜇 = 0, 𝜎² = 1)，截断到 [−3, 3]。
- **RMSNorm**：𝟙（即初始化为 1）

你应该使用 `torch.nn.init.trunc_normal_` 来初始化截断正态权重。

#### 3.3.2 Linear 模块

线性层是 Transformer 以及一般神经网络的基本构建块。首先，你将实现自己的 `Linear` 类，它继承自 `torch.nn.Module` 并执行线性变换：

𝑦 = 𝑊𝑥.    (3)

注意，遵循大多数现代 LLM 的做法，我们**不包含偏置项**。

**问题（linear）：实现 linear 模块（1 分）**

**交付物：** 实现一个继承自 `torch.nn.Module` 并执行线性变换的 `Linear` 类。你的实现应遵循 PyTorch 内置 `nn.Linear` 模块的接口，只是没有 bias 参数或参数。我们推荐以下接口：

```
def __init__(self, in_features, out_features, device=None, dtype=None)
```

构造一个线性变换模块。此函数应接受以下参数：

- `in_features: int` —— 输入的最终维度
- `out_features: int` —— 输出的最终维度
- `device: torch.device | None = None` —— 存储参数的设备
- `dtype: torch.dtype | None = None` —— 参数的数据类型

```
def forward(self, x: torch.Tensor) -> torch.Tensor
```

对输入应用线性变换。

请务必：

- 继承 `nn.Module`
- 调用父类构造函数
- 将你的参数构造并存储为 𝑊（而非 𝑊⊤），将其放入 `nn.Parameter` 中
- 当然，不要使用 `nn.Linear` 或 `nn.functional.linear`

对于初始化，使用上述设置以及 `torch.nn.init.trunc_normal_` 来初始化权重。

要测试你的 Linear 模块，实现位于 `[adapters.run_linear]` 的测试适配器。该适配器应把给定的权重加载到你的 Linear 模块中。你可以使用 `Module.load_state_dict` 来达到此目的。然后运行 `uv run pytest -k test_linear`。

#### 3.3.3 Embedding 模块

如上所述，Transformer 的第一层是一个嵌入层，它把整数 token ID 映射到维度为 d_model 的向量空间。我们将实现一个自定义的 `Embedding` 类，继承自 `torch.nn.Module`（所以你不应使用 `nn.Embedding`）。`forward` 方法应通过在形状为 `(vocab_size, d_model)` 的嵌入矩阵中索引，为每个 token ID 选择嵌入向量，使用形状为 `(batch_size, sequence_length)` 的 token ID 的 `torch.LongTensor`。

**问题（embedding）：实现 embedding 模块（1 分）**

**交付物：** 实现继承自 `torch.nn.Module` 并执行嵌入查找的 `Embedding` 类。你的实现应遵循 PyTorch 内置 `nn.Embedding` 模块的接口。我们推荐以下接口：

```
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
```

构造一个嵌入模块。此函数应接受以下参数：

- `num_embeddings: int` —— 词表大小
- `embedding_dim: int` —— 嵌入向量的维度，即 d_model
- `device: torch.device | None = None` —— 存储参数的设备
- `dtype: torch.dtype | None = None` —— 参数的数据类型

```
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
```

为给定的 token ID 查找嵌入向量。

请务必：

- 继承 `nn.Module`
- 调用父类构造函数
- 将你的嵌入矩阵初始化为 `nn.Parameter`
- 存储嵌入矩阵时，d_model 作为最后一个维度
- 当然，不要使用 `nn.Embedding` 或 `nn.functional.embedding`

再次使用上述设置进行初始化，并用 `torch.nn.init.trunc_normal_` 初始化权重。

要测试你的实现，实现位于 `[adapters.run_embedding]` 的测试适配器。然后运行 `uv run pytest -k test_embedding`。

### 3.4 Pre-Norm Transformer 块

每个 Transformer 块有两个子层：一个多头自注意力机制和一个逐位置前馈网络（[A. Vaswani et al., 2017] 第 3.1 节）。

在原始 Transformer 论文中，模型围绕两个子层各使用一个残差连接，其后接层归一化。这种架构通常被称为"post-norm"Transformer，因为层归一化应用于子层输出。然而，大量工作发现，把层归一化从每个子层的输出移到每个子层的输入（并在最后一个 Transformer 块之后再额外加一个层归一化）能提高 Transformer 训练稳定性 [T. Q. Nguyen et al., 2019; R. Xiong et al., 2020]——见图 2 对这种"pre-norm"Transformer 块的可视化表示。然后，每个 Transformer 块子层的输出通过残差连接加到子层输入上（A. Vaswani et al. [8] 第 5.4 节）。对 pre-norm 的一种直觉是：从输入嵌入到 Transformer 最终输出之间存在一条干净的、没有任何归一化的"残差流"（residual stream），据称这可以改善梯度流动。这种 pre-norm Transformer 现在已成为当今语言模型的标准（例如 GPT-3、LLaMA、PaLM 等），所以我们将实现这个变体。我们将逐一讲解 pre-norm Transformer 块的各个组件，并按顺序实现它们。

#### 3.4.1 均方根层归一化（RMSNorm）

A. Vaswani et al. [8] 的原始 Transformer 实现使用层归一化 [J. L. Ba et al., 2016] 来归一化激活值。遵循 H. Touvron et al. [12]，我们将使用均方根层归一化（RMSNorm；B. Zhang et al. [13] 公式 4）作为层归一化。给定一个激活值向量 𝑎 ∈ ℝ^{d_model}，RMSNorm 将按下式对每个激活值 𝑎_𝑖 进行缩放：

RMSNorm(𝑎_𝑖) = (𝑎_𝑖 / RMS(𝑎)) · 𝑔_𝑖,    (4)

其中 RMS(𝑎) = √( (1/d_model) Σ_{i=1}^{d_model} 𝑎_𝑖² + 𝜀 )。这里 𝑔_𝑖 是一个可学习的"增益"（gain）参数（总共有 d_model 个这样的参数），𝜀 是一个通常固定为 1e-5 的超参数。

你应该把输入上转换（upcast）到 `torch.float32`，以防止对输入平方时溢出。总体而言，你的 `forward` 方法应如下所示：

```
  in_dtype = x.dtype
  x = x.to(torch.float32)
  # 在这里写执行 RMSNorm 的代码
  ...
  result = ...
  # 以原始 dtype 返回结果
  return result.to(in_dtype)
```

**问题（rmsnorm）：均方根层归一化（1 分）**

**交付物：** 将 RMSNorm 实现为一个 `torch.nn.Module`。我们推荐以下接口：

```
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
```

构造 RMSNorm 模块。此函数应接受以下参数：

- `d_model: int` —— 模型的隐藏维度
- `eps: float = 1e-5` —— 数值稳定性的 epsilon 值
- `device: torch.device | None = None` —— 存储参数的设备
- `dtype: torch.dtype | None = None` —— 参数的数据类型

```
def forward(self, x: torch.Tensor) -> torch.Tensor
```

处理形状为 `(batch_size, sequence_length, d_model)` 的输入张量，并返回相同形状的张量。

注意：如上所述，在执行归一化之前请记得把输入上转换到 `torch.float32`（之后再下转换回原始 dtype）。

要测试你的实现，实现位于 `[adapters.run_rmsnorm]` 的测试适配器。然后运行 `uv run pytest -k test_rmsnorm`。

#### 3.4.2 逐位置前馈网络

> **图 3**：比较 SiLU（又称 Swish）与 ReLU 激活函数。
>
> SiLU: f(x) = x·σ(x)；Identity: f(x) = x；ReLU: f(x) = max(0, x)

在原始 Transformer 论文（A. Vaswani et al. [8] 第 3.3 节）中，Transformer 前馈网络由两个线性变换组成，中间夹着一个 ReLU 激活（ReLU(𝑥) = max(0,𝑥)）。在那个原始架构中，内部前馈层的维度通常是输入维度的 4 倍。

然而，现代语言模型相比这个原始设计倾向于纳入两个主要变化：它们使用另一种激活函数，并采用一种门控机制。具体来说，我们将实现 Llama 3 [A. Grattafiori et al., 2024] 和 Qwen 2.5 [A. Yang et al., 2024] 等 LLM 采用的"SwiGLU"激活函数，它把 SiLU（通常称为 Swish）激活与一种称为门控线性单元（Gated Linear Unit，GLU）的门控机制结合起来。我们还将省略线性层中有时使用的偏置项，遵循 PaLM [A. Chowdhery et al., 2022] 和 LLaMA [H. Touvron et al., 2023] 以来大多数现代 LLM 的做法。

SiLU 或 Swish 激活函数 [D. Hendrycks et al., 2016; S. Elfwing et al., 2017] 定义如下：

SiLU(𝑥) = 𝑥·𝜎(𝑥) = 𝑥/(1+𝑒^(−𝑥))    (5)

如图 3 所示，SiLU 激活函数与 ReLU 激活函数相似，但在零点处是平滑的。

门控线性单元（GLU）最初由 Y. N. Dauphin et al. [19] 定义为一个经过 sigmoid 函数的线性变换与另一个线性变换的逐元素乘积：

GLU(𝑥, 𝑊1, 𝑊2) = 𝜎(𝑊1𝑥) ⊙ 𝑊2𝑥,    (6)

其中 ⊙ 表示逐元素乘法。门控线性单元据称可以通过"为梯度提供一条线性路径，同时保留非线性能力"，从而"缓解深层架构的梯度消失问题"。

把 SiLU/Swish 与 GLU 结合起来，我们得到 SwiGLU，我们将把它用于我们的前馈网络：

FFN(𝑥) = SwiGLU(𝑥, 𝑊1, 𝑊2, 𝑊3) = 𝑊2 ( SiLU(𝑊1𝑥) ⊙ 𝑊3𝑥 ),    (7)

其中 𝑥 ∈ ℝ^{d_model}，𝑊1, 𝑊3 ∈ ℝ^{d_ff×d_model}，𝑊2 ∈ ℝ^{d_model×d_ff}，并且按惯例 d_ff = (8/3)·d_model。对于具体实现，为了硬件效率，将其四舍五入到最接近的 64 的倍数是没问题的。

N. Shazeer [20] 首先提出将 SiLU/Swish 激活与 GLU 结合，并做了实验表明 SwiGLU 在语言建模任务上优于 ReLU 和 SiLU（无门控）等基线。在作业的后续部分，你将比较 SwiGLU 与 SiLU。虽然我们已经提到了一些关于这些组件的启发式论证（论文提供了更多支持证据），但保持经验视角是好的：Shazeer 论文中有一句如今很出名的话：

> "我们并不解释为什么这些架构似乎有效；我们把它们的成功——一如其他一切——归功于神明的仁慈。"

**问题（positionwise_feedforward）：实现逐位置前馈网络（2 分）**

**交付物：** 实现 SwiGLU 前馈网络，由一个 SiLU 激活函数和一个 GLU 组成。

注意：在这种特定情况下，为了数值稳定性，你可以放心地在实现中使用 `torch.sigmoid`。

在你的实现中，你应把 d_ff 设为约 (8/3)×d_model，同时确保内部前馈层的维度是 64 的倍数，以充分利用硬件。要针对我们提供的测试来测试你的实现，你需要实现位于 `[adapters.run_swiglu]` 的测试适配器。然后运行 `uv run pytest -k test_swiglu` 来测试你的实现。

#### 3.4.3 相对位置嵌入（RoPE）

为了向模型注入位置信息，我们将实现旋转位置嵌入（Rotary Position Embeddings）[J. Su et al., 2021]，通常称为 RoPE。对于 token 位置 𝑖 处的给定查询 token 𝑞(𝑖) = 𝑊_𝑞 𝑥(𝑖) ∈ ℝ^𝑑，我们将应用一个成对旋转矩阵 𝑅_𝑖，得到 𝑞′(𝑖) = 𝑅_𝑖 𝑞(𝑖) = 𝑅_𝑖 𝑊_𝑞 𝑥(𝑖)。这里 𝑅_𝑖 将把嵌入元素对 𝑞(𝑖)_{2𝑘−1:2𝑘} 作为 2d 向量，旋转角度 𝜃_{𝑖,𝑘} = 𝑖·Θ^{(2𝑘−2)/𝑑}，其中 𝑘 ∈ {1,…,𝑑/2}，Θ 是某个常数。因此，我们可以把 𝑅_𝑖 看作一个大小为 𝑑×𝑑 的块对角矩阵，块为 𝑅_𝑖𝑘（𝑘 ∈ {1,…,𝑑/2}），其中

𝑅_𝑖𝑘 = ( cos(𝜃_{𝑖,𝑘})  −sin(𝜃_{𝑖,𝑘}) ; sin(𝜃_{𝑖,𝑘})  cos(𝜃_{𝑖,𝑘}) )    (8)

于是我们得到完整的旋转矩阵

𝑅_𝑖 = ( 𝑅_𝑖1 0 0 ⋯ 0 ; 0 𝑅_𝑖2 0 ⋯ 0 ; 0 0 𝑅_𝑖3 ⋯ 0 ; ⋮ ⋮ ⋮ ⋱ ⋮ ; 0 0 0 ⋯ 𝑅_𝑖(𝑑/2) ),    (9)

其中 0 表示 2×2 零矩阵。虽然可以构造完整的 𝑑×𝑑 矩阵，但好的解法应该利用该矩阵的性质来更高效地实现这个变换。由于我们只关心给定序列内 token 的**相对**旋转，我们可以跨层、跨不同批量复用在 cos(𝜃_{𝑖,𝑘}) 和 sin(𝜃_{𝑖,𝑘}) 上计算的值。如果你想优化它，可以使用一个被所有层引用的单一 RoPE 模块，它可以在 `__init__` 期间用 `self.register_buffer(persistent=False)` 创建一个 2d 的、预计算的 sin 和 cos 值缓冲区，而不是 `nn.Parameter`（因为我们不想学习这些固定的余弦和正弦值）。我们对 𝑞(𝑖) 所做的完全相同的旋转过程，也对 𝑘(𝑗) 执行，按相应的 𝑅_𝑗 旋转。注意，这一层没有可学习参数。

**问题（rope）：实现 RoPE（2 分）**

**交付物：** 实现一个对输入张量应用 RoPE 的 `RotaryPositionalEmbedding` 类。

推荐以下接口：

```
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
```

构造 RoPE 模块，并在需要时创建缓冲区。

- `theta: float` —— RoPE 的 Θ 值
- `d_k: int` —— 查询和键向量的维度
- `max_seq_len: int` —— 将被输入的最大序列长度
- `device: torch.device | None = None` —— 存储缓冲区的设备

```
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```

处理形状为 `(..., seq_len, d_k)` 的输入张量，并返回相同形状的张量。注意，你应该容忍 𝑥 具有任意数量的批量维度。你应假设 token 位置是形状为 `(..., seq_len)` 的张量，指定 𝑥 沿序列维度的 token 位置。

你应该使用 token 位置沿序列维度切片你的（可能是预计算的）cos 和 sin 张量。

要测试你的实现，完成 `[adapters.run_rope]` 并确保它通过 `uv run pytest -k test_rope`。

#### 3.4.4 缩放点积注意力

现在我们将实现 A. Vaswani et al. [8]（第 3.2.1 节）中描述的缩放点积注意力。作为预备步骤，Attention 操作的定义将使用 softmax，这是一个把未归一化的分数向量转换为归一化分布的操作：

softmax(𝑣)_𝑖 = exp(𝑣_𝑖) / Σ_{j=1}^n exp(𝑣_𝑗).    (10)

注意，exp(𝑣_𝑖) 对于较大的值可能变成 inf（于是 inf/inf = NaN）。我们可以通过注意到 softmax 操作对给所有输入加上任意常数 𝑐 是不变的来避免这一点。我们可以利用这个性质来实现数值稳定——通常，我们会从 𝑣 的所有元素中减去 𝑣 的最大项，使新的最大项变为 0。你现在将实现 softmax，并使用这个技巧来实现数值稳定。

**问题（softmax）：实现 softmax（1 分）**

**交付物：** 编写一个函数，对张量应用 softmax 操作。你的函数应接受两个参数：一个张量和一个维度 𝑖，并对输入张量的第 𝑖 个维度应用 softmax。输出张量应与输入张量形状相同，但其第 𝑖 个维度现在是一个归一化的概率分布。使用"从第 𝑖 个维度的所有元素中减去该维度的最大值"的技巧来避免数值稳定性问题。

要测试你的实现，完成 `[adapters.run_softmax]` 并确保它通过 `uv run pytest -k test_softmax_matches_pytorch`。

现在我们可以把 Attention 操作数学地定义如下：

Attention(𝑄, 𝐾, 𝑉) = softmax( 𝑄𝐾^𝑇 / √d_k ) 𝑉    (11)

其中 𝑄 ∈ ℝ^{n×d_k}，𝐾 ∈ ℝ^{m×d_k}，𝑉 ∈ ℝ^{m×d_v}。这里 𝑄、𝐾 和 𝑉 都是此操作的输入——注意这些不是可学习参数。

**掩码：** 有时对注意力操作的输出进行掩码会很方便。一个掩码应具有形状 𝑀 ∈ {True, False}^{n×m}，这个布尔矩阵的每一行 𝑖 表示查询 𝑖 应该关注哪些键。按惯例（且略微令人困惑），位置 (𝑖,𝑗) 处的 True 值表示查询 𝑖 **确实**关注键 𝑗，False 值表示查询不关注该键。换句话说，在 True 的 (𝑖,𝑗) 对处有"信息流动"。例如，考虑一个 1×3 掩码矩阵，其条目为 `[[True, True, False]]`。这个唯一的查询向量只关注前两个键。

在计算上，使用掩码比对子序列分别计算注意力要高效得多，我们可以这样做：取 softmax 之前的值（𝑄𝐾⊤/√d_k），对掩码矩阵中为 False 的任何条目加上 −∞。

**问题（scaled_dot_product_attention）：实现缩放点积注意力（5 分）**

**交付物：** 实现缩放点积注意力函数。你的实现应处理形状为 `(batch_size, ..., seq_len, d_k)` 的键和查询，以及形状为 `(batch_size, ..., seq_len, d_v)` 的值，其中 `...` 表示任意数量的其他类批量维度（如果提供）。实现应返回形状为 `(batch_size, ..., seq_len, d_v)` 的输出。关于类批量维度的讨论见 3.2 节。

你的实现还应支持可选的、用户提供的、形状为 `(seq_len, seq_len)` 的布尔掩码。掩码值为 True 的位置的注意力概率之和应为 1，掩码值为 False 的位置的注意力概率应为零。

要针对我们提供的测试来测试你的实现，你需要实现位于 `[adapters.run_scaled_dot_product_attention]` 的测试适配器。`uv run pytest -k test_scaled_dot_product_attention` 在三阶输入张量上测试你的实现，而 `uv run pytest -k test_4d_scaled_dot_product_attention` 在四阶输入张量上测试你的实现。

#### 3.4.5 因果多头自注意力

我们将实现 A. Vaswani et al. [8] 第 3.2.2 节中描述的多头自注意力。回顾一下，数学上应用多头注意力的操作定义如下：

MultiHead(𝑄, 𝐾, 𝑉) = Concat(head_1, …, head_ℎ)    (12)

其中

head_𝑖 = Attention(𝑄_𝑖, 𝐾_𝑖, 𝑉_𝑖)    (13)

其中 𝑄_𝑖、𝐾_𝑖、𝑉_𝑖 分别是 𝑄、𝐾、𝑉 的嵌入维度中大小为 d_k 或 d_v 的第 𝑖 ∈ {1,…,ℎ} 个切片。Attention 是 3.4.4 节定义的缩放点积注意力操作。由此我们可以构造多头自注意力操作：

MultiHeadSelfAttention(𝑥) = 𝑊_𝑂 · MultiHead(𝑊_𝑄𝑥, 𝑊_𝐾𝑥, 𝑊_𝑉𝑥)    (14)

这里可学习参数是 𝑊_𝑄 ∈ ℝ^{ℎd_k×d_model}，𝑊_𝐾 ∈ ℝ^{ℎd_k×d_model}，𝑊_𝑉 ∈ ℝ^{ℎd_v×d_model}，以及 𝑊_𝑂 ∈ ℝ^{d_model×ℎd_v}。由于 𝑄、𝐾、𝑉 在多头注意力操作中被切片，我们可以把 𝑊_𝑄、𝑊_𝐾、𝑊_𝑉 看作沿输出维度为每个头分开。当你把这做好后，你应该用总共三次矩阵乘法来计算键、值和查询投影。⁵

⁵作为一个拓展目标，尝试把键、查询和值投影合并成一个权重矩阵，这样你只需要一次矩阵乘法。

**因果掩码**

你的实现应防止模型关注序列中的未来 token。换句话说，如果给模型一个 token 序列 𝑡_1,…,𝑡_𝑛，我们想为前缀 𝑡_1,…,𝑡_𝑖（其中 𝑖 < 𝑛）计算下一个词预测，那么模型不应能访问（关注）位置 𝑡_{𝑖+1},…,𝑡_𝑛 处的 token 表示，因为在推理生成文本时，它无法访问这些 token（而且这些未来 token 会泄露关于真实下一个词身份的信息，使语言建模预训练目标变得平凡）。对于输入 token 序列 𝑡_1,…,𝑡_𝑛，我们可以通过运行 𝑛 次多头自注意力（对应序列中的 𝑛 个唯一前缀）来朴素地防止访问未来 token。相反，我们将使用**因果注意力掩码**，它允许 token 𝑖 关注序列中所有位置 𝑗 ≤ 𝑖。你可以使用 `torch.triu` 或广播的索引比较来构造这个掩码，并且你应该利用你 3.4.4 节的缩放点积注意力实现已经支持注意力掩码这一事实。

**应用 RoPE**

RoPE 应应用于查询和键向量，但不应用于值向量。此外，头维度应作为批量维度处理，因为在多头注意力中，注意力是独立应用于每个头的。这意味着对每个头的查询和键向量应应用完全相同的 RoPE 旋转。

**问题（multihead_self_attention）：实现因果多头自注意力（5 分）**

**交付物：** 将因果多头自注意力实现为一个 `torch.nn.Module`。你的实现应接受（至少）以下参数：

- `d_model: int` —— Transformer 块输入的维度。
- `num_heads: int` —— 多头自注意力中使用的头数。

遵循 A. Vaswani et al. [8]，设 d_k = d_v = d_model/ℎ。要针对我们提供的测试来测试你的实现，实现位于 `[adapters.run_multihead_self_attention]` 的测试适配器。然后运行 `uv run pytest -k test_multihead_self_attention` 来测试你的实现。

### 3.5 完整的 Transformer LM

让我们先把 Transformer 块组装起来（回头参考图 2 会很有帮助）。一个 Transformer 块包含两个"子层"，一个用于多头自注意力，另一个用于 SwiGLU 前馈网络。在每个子层中，我们首先执行 RMSNorm，然后执行主要操作（MHA/FF），最后加上残差连接。

具体来说，Transformer 块的前半部分（第一个"子层"）应实现以下一组更新，以从输入 𝑥 产生输出 𝑦，

𝑦 = 𝑥 + MultiHeadSelfAttention(RMSNorm(𝑥)).    (15)

**问题（transformer_block）：实现 Transformer 块（3 分）**

实现 3.4 节描述、图 2 展示的 pre-norm Transformer 块。你的 Transformer 块应接受（至少）以下参数。

- `d_model: int` —— Transformer 块输入的维度。
- `num_heads: int` —— 多头自注意力中使用的头数。
- `d_ff: int` —— 逐位置前馈内部层的维度。

要测试你的实现，实现适配器 `[adapters.run_transformer_block]`。然后运行 `uv run pytest -k test_transformer_block` 来测试你的实现。

**交付物：** 通过所提供测试的 Transformer 块代码。

现在我们把这些块组合起来，遵循图 1 的高层示意图。遵循我们在 3.1 节中对嵌入的描述，把它送入 `num_layers` 个 Transformer 块，然后再送入最终的层归一化和 LM head，以获得词表上的未归一化分布（logits）。

**问题（transformer_lm）：实现 Transformer LM（3 分）**

是时候把所有东西组合起来了！实现 3.1 节描述、图 1 展示的 Transformer 语言模型。你的实现至少应接受上述 Transformer 块的所有构造参数，以及以下附加参数：

- `vocab_size: int` —— 词表大小，用于确定 token 嵌入矩阵的维度。
- `context_length: int` —— 最大上下文长度，用于确定 RoPE sin 和 cos 缓冲区的维度。
- `num_layers: int` —— 使用的 Transformer 块数。

要针对我们提供的测试来测试你的实现，你首先需要实现位于 `[adapters.run_transformer_lm]` 的测试适配器。然后运行 `uv run pytest -k test_transformer_lm` 来测试你的实现。

**交付物：** 通过上述测试的 Transformer LM 模块。

**资源核算**

能够理解 Transformer 的各个部分如何消耗计算和内存是很有用的。我们将逐步做一些基本的"FLOPs 核算"。Transformer 中绝大多数的 FLOPs 是矩阵乘法，所以我们的核心方法很简单：

1. 写下 Transformer 前向传播中的所有矩阵乘法。
2. 把每个矩阵乘法转换为所需的 FLOPs。

对于第二步，以下事实会很有用：

**规则：** 给定 𝐴 ∈ ℝ^{m×n} 和 𝐵 ∈ ℝ^{n×p}，矩阵-矩阵乘积 𝐴𝐵 需要 2𝑚𝑛𝑝 FLOPs。

为了理解这一点，注意 (𝐴𝐵)[𝑖,𝑗] = 𝐴[𝑖,:]·𝐵[:,𝑗]，这个点积需要 𝑛 次加法和 𝑛 次乘法（2𝑛 FLOPs）。然后，由于矩阵-矩阵乘积 𝐴𝐵 有 𝑚×𝑝 个条目，总 FLOPs 数为 (2𝑛)(𝑚𝑝) = 2𝑚𝑛𝑝。

现在，在做下一个问题之前，逐个过一遍你 Transformer 块和 Transformer LM 的每个组件，列出所有矩阵乘法及其相关 FLOPs 成本，会很有帮助。

**问题（transformer_accounting）：Transformer LM 资源核算（5 分）**

(a) 考虑一个使用我们作业架构的 GPT-2 XL 规模模型，其配置如下：

- `vocab_size:` 50,257
- `context_length:` 1,024
- `num_layers:` 48
- `d_model:` 1,600
- `num_heads:` 25
- `d_ff:` 4,288（即最接近 (8/3)×1,600 的 64 的倍数）

假设我们用这个配置构造模型。我们的模型会有多少可训练参数？假设每个参数用单精度浮点数表示，仅加载这个模型需要多少内存？

**交付物：** 一到两句话的回答。

(b) 找出完成我们 GPT-2 XL 形状模型的一次前向传播所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs？假设我们的输入序列有 context_length 个 token。

**交付物：** 一个矩阵乘法列表（带描述），以及所需的总 FLOPs 数。

(c) 基于你上面的分析，模型的哪些部分需要最多的 FLOPs？

**交付物：** 一到两句话的回答。

(d) 用 GPT-2 small（12 层，768 d_model，12 头）、GPT-2 medium（24 层，1024 d_model，16 头）和 GPT-2 large（36 层，1280 d_model，20 头）重复你的分析。随着模型规模增大，Transformer LM 的哪些部分在总 FLOPs 中占的比例增加或减少？

**交付物：** 对每个模型，给出模型组件的分解及其相关 FLOPs（作为一次前向传播所需总 FLOPs 的比例）。此外，用一到两句话描述改变模型规模如何改变每个组件的 FLOPs 占比。

(e) 取 GPT-2 XL 并把上下文长度增加到 16,384。一次前向传播的总 FLOPs 如何变化？各模型组件的 FLOPs 相对贡献如何变化？

**交付物：** 一到两句话的回答。

## 4 训练 Transformer LM

我们现在已经有了预处理数据的步骤（通过分词器）和模型（Transformer）。剩下的是构建支持训练所需的所有代码。这包括以下内容：

- **损失**：我们需要定义损失函数（交叉熵）。
- **优化器**：我们需要定义最小化该损失的优化器（AdamW）。
- **训练循环**：我们需要所有加载数据、保存检查点和管理训练的支撑性基础设施。

### 4.1 交叉熵损失

回顾一下，Transformer 语言模型为每个长度为 𝑚+1 的序列 𝑥 和 𝑖 = 1,…,𝑚 定义了一个分布 𝑝_𝜃(𝑥_{𝑖+1} | 𝑥_{1:𝑖})。给定一个由长度为 𝑚+1 的序列组成的训练集 𝐷，我们定义标准的交叉熵（负对数似然）损失函数：

ℓ(𝜃; 𝐷) = (1/|𝐷|)·(1/𝑚)·Σ_{𝑥∈𝐷} Σ_{𝑖=1}^{𝑚} −log 𝑝_𝜃(𝑥_{𝑖+1} | 𝑥_{1:𝑖}).    (16)

（注意，Transformer 中的一次前向传播就为所有 𝑖 = 1,…,𝑚 产生了 𝑝_𝜃(𝑥_{𝑖+1} | 𝑥_{1:𝑖})。）

特别地，Transformer 为每个位置 𝑖 计算 logits 𝑜_𝑖 ∈ ℝ^{vocab_size}，从而得到：⁶

𝑝(𝑥_{𝑖+1} | 𝑥_{1:𝑖}) = softmax(𝑜_𝑖)[𝑥_{𝑖+1}] = exp(𝑜_𝑖[𝑥_{𝑖+1}]) / Σ_{𝑎=1}^{vocab_size} exp(𝑜_𝑖[𝑎]).    (17)

交叉熵损失通常相对于 logits 向量 𝑜_𝑖 ∈ ℝ^{vocab_size} 和目标 𝑥_{𝑖+1} 来定义。⁷

⁶注意 𝑜_𝑖[𝑘] 指的是向量 𝑜_𝑖 在索引 𝑘 处的值。

⁷这对应于在 𝑥_{𝑖+1} 上的 Dirac delta 分布与预测的 softmax(𝑜_𝑖) 分布之间的交叉熵。

实现交叉熵损失需要像 softmax 的情况一样注意数值问题。

**问题（cross_entropy）：实现交叉熵（1 分）**

**交付物：** 编写一个函数来计算交叉熵损失，它接收预测的 logits（𝑜_𝑖）和目标（𝑥_{𝑖+1}），并计算交叉熵 ℓ_𝑖 = −log softmax(𝑜_𝑖)[𝑥_{𝑖+1}]。你的函数应处理以下情况：

- 减去最大元素以保证数值稳定。
- 尽可能抵消 log 和 exp。
- 处理任何额外的批量维度，并返回跨批量的平均值。与 3.2 节一样，我们假设类批量维度总是排在前面，位于词表大小维度之前。

实现 `[adapters.run_cross_entropy]`，然后运行 `uv run pytest -k test_cross_entropy` 来测试你的实现。

**困惑度（Perplexity）**

交叉熵对于训练已经足够，但当我们评估模型时，我们还希望报告困惑度。对于一个长度为 𝑚 的序列，其交叉熵损失为 ℓ_1,…,ℓ_𝑚：

perplexity = exp( (1/𝑚) Σ_{𝑖=1}^{𝑚} ℓ_𝑖 ).    (18)

### 4.2 SGD 优化器

现在我们有了损失函数，就开始探索优化器。最简单的基于梯度的优化器是随机梯度下降（SGD）。我们从随机初始化的参数 𝜃_0 开始。然后对于每一步 𝑡 = 0,…,𝑇−1，我们执行以下更新：

𝜃_{𝑡+1} ← 𝜃_𝑡 − 𝛼_𝑡 ∇𝐿(𝜃_𝑡; 𝐵_𝑡),    (19)

其中 𝐵_𝑡 是从数据集 𝐷 中采样的一批随机数据，学习率 𝛼_𝑡 和批量大小 |𝐵_𝑡| 是超参数。

#### 4.2.1 在 PyTorch 中实现 SGD

要实现我们的优化器，我们将继承 PyTorch 的 `torch.optim.Optimizer` 类。一个 Optimizer 子类必须实现两个方法：

- `def __init__(self, params, ...)` 应初始化你的优化器。这里 `params` 将是一组要优化的参数（或参数组，以防用户想对模型的不同部分使用不同的超参数，例如不同的学习率）。请务必把 `params` 传给基类的 `__init__` 方法，它会存储这些参数供 `step` 使用。你可以根据优化器接受额外参数（例如学习率是常见的一个），并把它们作为字典传给基类构造函数，其中键是你为这些参数选择的名字（字符串）。

- `def step(self)` 应对参数做一次更新。在训练循环中，它会在反向传播之后被调用，所以你可以访问最后一个批量上的梯度。这个方法应遍历每个参数张量 p 并就地修改它们，即设置 `p.data`（它持有与该参数关联的张量），基于梯度 `p.grad`（如果存在的话）——它表示损失关于该参数的梯度张量。

PyTorch 优化器 API 有一些微妙之处，所以用一个例子来解释会更容易。为了让我们的例子更丰富，我们实现一个略有变化的 SGD，其中学习率随训练衰减，从初始学习率 𝛼 开始，随时间采取越来越小的步长：

𝜃_{𝑡+1} = 𝜃_𝑡 − (𝛼/√(𝑡+1)) ∇𝐿(𝜃_𝑡; 𝐵_𝑡)    (20)

让我们看看这个版本的 SGD 如何实现为一个 PyTorch Optimizer：

```
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率。
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # 获取与 p 关联的状态。
                t = state.get("t", 0)  # 从状态中获取迭代次数，或 0。
                grad = p.grad.data  # 获取损失关于 p 的梯度。
                p.data -= lr / math.sqrt(t + 1) * grad  # 就地更新权重张量。
                state["t"] = t + 1  # 递增迭代次数。
        return loss
```

在 `__init__` 中，我们把参数以及默认超参数传给基类构造函数（参数可能分组出现，每组有不同的超参数）。如果参数只是单个 `torch.nn.Parameter` 对象的集合，基类构造函数会创建一个组并给它分配默认超参数。然后，在 `step` 中，我们遍历每个参数组，再遍历该组中的每个参数，应用公式 20。这里我们把迭代次数作为与每个参数关联的状态保存：我们先读取该值，在梯度更新中使用它，然后更新它。API 规定用户可以传入一个可调用的 `closure`，在优化器步进之前重新计算损失。对于我们将使用的优化器，我们不需要这个，但我们加上它以符合 API。

为了看看它是否有效，我们可以使用下面这个最小训练循环示例：

```
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)
for t in range(100):
    opt.zero_grad()  # 重置所有可学习参数的梯度。
    loss = (weights**2).mean() # 计算一个标量损失值。
    print(loss.cpu().item())
    loss.backward() # 运行反向传播，计算梯度。
    opt.step() # 运行优化器步进。
```

这是训练循环的典型结构：在每次迭代中，我们计算损失并运行一步优化器。在训练语言模型时，我们的可学习参数来自模型（在 PyTorch 中，`m.parameters()` 给我们这个集合）。损失将在一个采样批量数据上计算，但训练循环的基本结构是相同的。

**问题（learning_rate_tuning）：调学习率（1 分）**

正如我们将看到的，对训练影响最大的超参数之一是学习率。让我们在这个玩具示例中实践一下。用另外三个学习率值运行上面的 SGD 示例：1e1、1e2 和 1e3，只训练 10 次迭代。对于这些学习率中的每一个，损失会怎样？它衰减得更快、更慢，还是发散（即在训练过程中增大）？

**交付物：** 用一到两句话描述你观察到的行为。

### 4.3 AdamW

现代语言模型通常使用更复杂的优化器训练，而不是 SGD。最近使用的大多数优化器都是 Adam 优化器 [D. P. Kingma et al., 2015] 的衍生品。我们将使用 AdamW [I. Loshchilov et al., 2019]，它在近期工作中被广泛使用。AdamW 提出了对 Adam 的一种修改，通过添加权重衰减（在每次迭代时把参数向 0 拉），以与梯度更新解耦的方式改善正则化。我们将按照 I. Loshchilov et al. [23] 的算法 2 中描述的那样实现 AdamW。

AdamW 是有状态的：对每个参数，它跟踪其一阶矩和二阶矩的运行估计。因此，AdamW 使用额外的内存来换取改善的稳定性和收敛性。除了学习率 𝛼，AdamW 还有一对超参数 (𝛽1, 𝛽2) 控制对矩估计的更新，以及一个权重衰减率 𝜆。典型应用把 (𝛽1, 𝛽2) 设为 (0.9, 0.999)，但 LLaMA [H. Touvron et al., 2023] 和 GPT-3 [T. B. Brown et al., 2020] 等大型语言模型常用 (0.9, 0.95) 训练。算法可以写如下，其中 𝜀 是一个小值（例如 10⁻⁸），用于在 𝑣 中出现极小值时改善数值稳定：

> **算法 1：AdamW 优化器**
>
> 1. init(𝜃) ▷ 初始化可学习参数
> 2. 𝑚 ← 0 ▷ 一阶矩向量的初始值；与 𝜃 同形状
> 3. 𝑣 ← 0 ▷ 二阶矩向量的初始值；与 𝜃 同形状
> 4. for 𝑡 = 1,…,𝑇 do
> 5. &nbsp;&nbsp;&nbsp;&nbsp;采样批量数据 𝐵_𝑡
> 6. &nbsp;&nbsp;&nbsp;&nbsp;𝑔 ← ∇_𝜃 ℓ(𝜃; 𝐵_𝑡) ▷ 计算损失的梯度
> 7. &nbsp;&nbsp;&nbsp;&nbsp;𝛼_𝑡 ← 𝛼·(√(1−𝛽_2^𝑡))/(1−𝛽_1^𝑡) ▷ 计算迭代 𝑡 调整后的 𝛼
> 8. &nbsp;&nbsp;&nbsp;&nbsp;𝜃 ← 𝜃 − 𝛼𝜆𝜃 ▷ 应用权重衰减
> 9. &nbsp;&nbsp;&nbsp;&nbsp;𝑚 ← 𝛽_1 𝑚 + (1−𝛽_1) 𝑔 ▷ 更新一阶矩估计
> 10. &nbsp;&nbsp;&nbsp;&nbsp;𝑣 ← 𝛽_2 𝑣 + (1−𝛽_2) 𝑔² ▷ 更新二阶矩估计
> 11. &nbsp;&nbsp;&nbsp;&nbsp;𝜃 ← 𝜃 − 𝛼_𝑡 · 𝑚/(√𝑣 + 𝜀) ▷ 应用矩调整后的权重更新
> 12. end for

注意 𝑡 从 1 开始。你现在将实现这个优化器。

**问题（adamw）：实现 AdamW（2 分）**

**交付物：** 将 AdamW 优化器实现为 `torch.optim.Optimizer` 的子类。你的类应在 `__init__` 中接受学习率 𝛼，以及 𝛽、𝜀 和 𝜆 超参数。为了帮助你保持状态，基类 Optimizer 给你一个字典 `self.state`，它把 `nn.Parameter` 对象映射到一个字典，该字典存储你需要的关于该参数的任何信息（对于 AdamW，这就是矩估计）。实现 `[adapters.get_adamw_cls]` 并确保它通过 `uv run pytest -k test_adamw`。

**问题（adamw_accounting）：使用 AdamW 训练的资源核算（2 分）**

让我们计算运行 AdamW 需要多少内存和计算。假设我们对每个张量都使用 float32。

(a) 运行 AdamW 需要多少峰值内存？根据参数、激活值、梯度和优化器状态的内存使用来分解你的答案。用 batch_size 和模型超参数（vocab_size、context_length、num_layers、d_model、num_heads）来表达你的答案。假设 d_ff = (8/3)×d_model。

为简单起见，在计算激活值的内存使用量时，只考虑以下组件：

- Transformer 块
  - RMSNorm(s)
  - 多头自注意力子层：𝑄𝐾𝑉 投影、𝑄𝐾⊤ 矩阵乘法、softmax、值的加权和、输出投影。
  - 逐位置前馈（SwiGLU）：𝑊1、𝑊2、门控分支上的 SiLU、逐元素乘积、𝑊3
- 最终 RMSNorm
- 输出嵌入
- logits 上的交叉熵

**交付物：** 对参数、激活值、梯度和优化器状态各自给出代数表达式，以及总计。

(b) 为 GPT-2 XL 形状的模型代入你的答案，得到一个只依赖于 batch_size 的表达式。你能使用的、在 80GB 内存内仍能容纳的最大批量大小是多少？

**交付物：** 一个形如 𝑎·batch_size + 𝑏（其中 𝑎、𝑏 为数值）的表达式，以及一个表示最大批量大小的数字。

(c) 运行一步 AdamW 需要多少 FLOPs？

**交付物：** 一个代数表达式，并附简短论证。

(d) 模型 FLOPs 利用率（MFU）定义为观测吞吐量（tokens/秒）与硬件理论峰值 FLOP 吞吐量的比值 [A. Chowdhery et al., 2022]。一块 NVIDIA H100 GPU 对 "float32"（实际上是 TensorFloat-32，在现实中是 "bfloat19"）操作的理论峰值为 495 teraFLOP/s。假设你能达到 50% 的 MFU，在单块 H100 上以批量大小 1024 训练 GPT-2 XL 400K 步需要多长时间？遵循 J. Kaplan et al. [25] 和 J. Hoffmann et al. [26]，假设反向传播的 FLOPs 是前向传播的两倍。

**交付物：** 训练所需的小时数，并附简短论证。

### 4.4 学习率调度

使损失下降最快的学习率值在训练过程中往往会变化。在训练 Transformer 时，通常使用学习率调度（schedule），即从较大的学习率开始，在开始时更快地更新，并随着模型训练缓慢衰减到较小的值。⁸ 在本作业中，我们将实现用于训练 LLaMA [H. Touvron et al., 2023] 的余弦退火调度。

调度器只是一个函数，它接收当前步 𝑡 和其他相关参数（如初始和最终学习率），并返回第 𝑡 步梯度更新要使用的学习率。最简单的调度是常数函数，对任何 𝑡 都返回相同的学习率。

余弦退火学习率调度接收 (i) 当前迭代 𝑡，(ii) 最大学习率 𝛼_max，(iii) 最小（最终）学习率 𝛼_min，(iv) 预热迭代数 𝑇_w，以及 (v) 余弦退火的最终迭代 𝑇_c。迭代 𝑡 处的学习率定义如下：

- （预热）如果 𝑡 < 𝑇_w，则 𝛼_𝑡 = (𝑡/𝑇_w)·𝛼_max。
- （余弦退火）如果 𝑇_w ≤ 𝑡 ≤ 𝑇_c，则 𝛼_𝑡 = 𝛼_min + (1/2)(1 + cos( ((𝑡−𝑇_w)/(𝑇_c−𝑇_w))·𝜋 ))(𝛼_max − 𝛼_min)。
- （退火后）如果 𝑡 > 𝑇_c，则 𝛼_𝑡 = 𝛼_min。

⁸有时也会使用学习率回升（restart）的调度，以帮助越过局部极小值。

**问题（learning_rate_schedule）：实现带预热的余弦学习率调度（1 分）**

编写一个函数，接收 𝑡、𝛼_max、𝛼_min、𝑇_w 和 𝑇_c，并根据上述调度器返回学习率 𝛼_𝑡。然后实现 `[adapters.get_lr_cosine_schedule]` 并确保它通过 `uv run pytest -k test_get_lr_cosine_schedule`。

### 4.5 梯度裁剪

在训练过程中，我们有时会遇到产生大梯度的训练样本，这可能破坏训练稳定性。为了缓解这一点，实践中常用的一种技术是梯度裁剪。其思想是，在每次反向传播之后、执行优化器步进之前，对梯度的范数施加一个限制。

给定（所有参数的）梯度 𝑔，我们计算它的 ℓ2-范数 ‖𝑔‖₂。如果这个范数小于最大值 𝑀，则保持 𝑔 不变；否则，我们把 𝑔 按因子 𝑀/(‖𝑔‖₂ + 𝜀) 缩小（其中加入一个小 𝜀，如 10⁻⁶，用于数值稳定）。注意，得到的范数将略小于 𝑀。

**问题（gradient_clipping）：实现梯度裁剪（1 分）**

编写一个实现梯度裁剪的函数。你的函数应接受一个参数列表和一个最大 ℓ2-范数。它应就地修改每个参数的梯度。使用 𝜀 = 10⁻⁶（PyTorch 默认值）。然后，实现适配器 `[adapters.run_gradient_clipping]` 并确保它通过 `uv run pytest -k test_gradient_clipping`。

## 5 训练循环

我们现在终于要把迄今为止构建的主要组件组合起来了：分词后的数据、模型和优化器。

### 5.1 数据加载器

分词后的数据（例如你在 tokenizer_experiments 中准备的数据）是一个单一的 token 序列 𝑥 = (𝑥_1,…,𝑥_𝑛)。即使源数据可能由独立的文档组成（例如不同的网页或源代码文件），一种常见做法是把它们全部拼接成一个 token 序列，在它们之间加一个分隔符（例如 `<|endoftext|>` token）。

数据加载器把它变成一个批量流，其中每个批量由 𝐵 个长度为 𝑚 的序列组成，并配对相应的下一个 token（长度也为 𝑚）。例如，对于 𝐵 = 1, 𝑚 = 3，`([𝑥_2, 𝑥_3, 𝑥_4], [𝑥_3, 𝑥_4, 𝑥_5])` 是一个可能的批量。

以这种方式加载数据简化了训练，原因有几个。首先，任何 1 ≤ 𝑖 ≤ 𝑛−𝑚 都给出一个有效的训练序列，所以采样训练序列是平凡的。由于所有训练序列长度相同，无需填充输入序列，这改善了硬件利用率（也通过增加批量大小 𝐵）。最后，我们也不需要加载完整数据集来采样训练数据，从而易于处理可能无法放入内存的大型数据集。

**问题（data_loading）：实现数据加载（2 分）**

**交付物：** 编写一个函数，接收一个 numpy 数组 𝑥（含 token ID 的整数数组）、一个 batch_size、一个 context_length 和一个 PyTorch 设备字符串（例如 `'cpu'` 或 `'cuda:0'`），并返回一对张量：采样出的输入序列和对应的下一个 token 目标。两个张量的形状都应为 `(batch_size, context_length)` 并包含 token ID，且两者都应放在所请求的设备上。要针对我们提供的测试来测试你的实现，你首先需要实现位于 `[adapters.run_get_batch]` 的测试适配器。然后运行 `uv run pytest -k test_get_batch` 来测试你的实现。

> **低资源提示：在 CPU 或 Apple Silicon 上加载数据**
>
> 如果你计划在 CPU 或 Apple Silicon 上训练 LM，你需要把数据移到正确的设备上（同样地，之后你也应该对模型使用相同的设备）。
>
> 如果你在 CPU 上，可以使用 `'cpu'` 设备字符串；在 Apple Silicon（M* 芯片）上，可以使用 `'mps'` 设备字符串。
>
> 关于 MPS 的更多信息，请查看以下资源：
> - https://docs.pytorch.org/docs/stable/mps.html
> - https://docs.pytorch.org/docs/stable/notes/mps.html
> - https://developer.apple.com/documentation/metalperformanceshaders

如果数据集太大无法加载到内存怎么办？我们可以使用名为 `mmap` 的 Unix 系统调用，它把磁盘上的文件映射到虚拟内存，并在访问该内存位置时惰性加载文件内容。因此，你可以"假装"整个数据集都在内存中。Numpy 通过 `np.memmap`（或对 `np.load` 使用标志 `mmap_mode='r'`，如果你最初用 `np.save` 保存数组的话）实现这一点，它会返回一个类似 numpy 数组的对象，在你访问条目时按需加载它们。在训练期间从你的数据集（即 numpy 数组）采样时，请务必以内存映射模式加载数据集（通过 `np.memmap` 或对 `np.load` 使用标志 `mmap_mode='r'`，取决于你保存数组的方式）。请确保还指定一个与你加载的数组匹配的 dtype。显式验证内存映射数据看起来正确（例如不包含超出预期词表大小的值）可能会很有帮助。

### 5.2 检查点（Checkpointing）

除了加载数据，我们还需要在训练时保存模型。在运行作业时，我们通常希望能够恢复中途停止的训练（例如由于作业超时、机器故障等）。即使一切顺利，我们可能也希望之后能访问中间模型（例如事后研究训练动态、从不同训练阶段取模型采样等）。

一个检查点应包含恢复训练所需的所有状态。我们当然至少希望能够恢复模型权重。如果使用有状态优化器（如 AdamW），我们还需要保存优化器的状态（例如对 AdamW 来说就是矩估计）。最后，为了恢复学习率调度，我们需要知道停在哪一次迭代。PyTorch 让保存所有这些变得容易：每个 `nn.Module` 都有一个 `state_dict()` 方法，返回一个包含所有可学习权重的字典；我们可以稍后用姊妹方法 `load_state_dict()` 恢复这些权重。任何 `torch.optim.Optimizer` 也是如此。最后，`torch.save(obj, dest)` 可以把一个对象（例如一个字典，其中一些值是张量，但也包含整数等普通 Python 对象）转储到文件（路径）或类文件对象，然后可以用 `torch.load(src)` 加载回内存。

**问题（checkpointing）：实现模型检查点（1 分）**

实现以下两个函数来加载和保存检查点：

```
def save_checkpoint(model, optimizer, iteration, out)
```

应把模型、优化器和迭代的所有状态转储到类文件对象 `out` 中。你可以使用模型和优化器的 `state_dict` 方法获取它们相关的状态，并使用 `torch.save(obj, out)` 把 `obj` 转储到 `out` 中（PyTorch 在这里支持路径或类文件对象）。一个典型的选择是让 `obj` 是一个字典，但只要你能稍后加载你的检查点，使用什么格式都可以。

此函数期望以下参数：

- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`
- `iteration: int`
- `out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`

```
def load_checkpoint(src, model, optimizer)
```

应从 `src`（路径或类文件对象）加载检查点，然后从该检查点恢复模型和优化器状态。你的函数应返回保存到检查点中的迭代次数。你可以使用 `torch.load(src)` 恢复你在 `save_checkpoint` 实现中保存的内容，并使用模型和优化器中的 `load_state_dict` 方法把它们恢复到之前的状态。

此函数期望以下参数：

- `src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`

实现 `[adapters.run_save_checkpoint]` 和 `[adapters.run_load_checkpoint]` 适配器，并确保它们通过 `uv run pytest -k test_checkpointing`。

### 5.3 训练循环

现在，终于到了把你实现的所有组件组合到你的主训练脚本中的时候了。让你能够轻松地用不同超参数启动训练（例如把它们作为命令行参数）会很有回报，因为你之后会多次这样做，以研究不同选择如何影响训练。

**问题（training_together）：组合起来（4 分）**

**交付物：** 编写一个脚本，运行训练循环，在用户提供的输入上训练你的模型。特别地，我们建议你的训练脚本至少允许：

- 配置和控制各种模型和优化器超参数的能力。
- 使用 `np.memmap` 内存高效地加载大型训练集和验证集。
- 把检查点序列化到用户提供的路径。
- 周期性记录训练和验证性能（例如记录到控制台和/或 Weights and Biases 等外部服务）。⁹

⁹wandb.ai

## 6 文本生成

现在我们可以训练模型了，最后一块拼图是从我们的模型生成文本的能力。回顾一下，语言模型接收一个（可能是批量的）长度为 sequence_length 的整数序列，并产生一个大小为 `(sequence_length, vocab_size)` 的矩阵，其中序列的每个元素都是一个预测该位置之后下一个 token 的概率分布。我们现在将编写一些函数，把它变成新序列的采样方案。

**Softmax**

按标准约定，语言模型的输出是最终线性层的输出（"logits"），所以我们必须通过 softmax 操作（我们之前在公式 10 中见过）把它变成归一化概率。

**解码**

要从我们的模型生成文本（解码），我们将给模型提供一串前缀 token（"提示词" prompt），并要求它产生一个预测序列中下一个 token 的在词表上的概率分布。然后，我们从这个在词表项上的分布采样，以确定下一个输出 token。

具体来说，解码过程的一步应接收一个序列 𝑥_{1…𝑡}，并通过以下公式返回一个 token 𝑥_{𝑡+1}，

𝑃(𝑥_{𝑡+1} = 𝑖 | 𝑥_{1…𝑡}) = exp(𝑣_𝑖) / Σ_𝑗 exp(𝑣_𝑗)    (21)

其中

𝑣 = TransformerLM(𝑥_{1…𝑡})_𝑡 ∈ ℝ^{vocab_size}    (22)

其中 TransformerLM 是我们的模型，它接收一个长度为 sequence_length 的序列作为输入，并产生一个大小为 `(sequence_length, vocab_size)` 的矩阵，我们取这个矩阵的最后一个元素，因为我们寻找的是第 𝑡 个位置处的下一个 token 预测。

这给了我们一个基本的解码器，通过反复从这些单步条件分布采样（把我们先前生成的输出 token 附加到下一个解码时间步的输入），直到我们生成序列结束 token `<|endoftext|>`（或用户指定的最大生成 token 数）。

**解码技巧**

我们将用小模型做实验，小模型有时会生成质量很低的文本。两个简单的解码技巧可以帮助解决这些问题。首先，在**温度缩放**中，我们用温度参数 𝜏 修改我们的 softmax，新的 softmax 为

softmax(𝑣, 𝜏)_𝑖 = exp(𝑣_𝑖/𝜏) / Σ_{𝑗=1}^{vocab_size} exp(𝑣_𝑗/𝜏).    (23)

注意，令 𝜏 → 0 会使 𝑣 的最大元素占主导，softmax 的输出变成一个集中在这个最大元素上的 one-hot 向量。

其次，另一个技巧是**核采样（nucleus）或 top-p 采样**，我们通过截断低概率 token 来修改采样分布。设 𝑞 是从（温度缩放的）大小为 vocab_size 的 softmax 得到的概率分布。超参数为 𝑝 的核采样按下式产生下一个 token：

```
            ⎧  q_i / Σ_{j∈V(p)} q_j   如果 i ∈ V(p)
P(x_{t+1}=i | q) = ⎨
            ⎩  0                      否则
```

其中 V(p) 是使 Σ_{j∈V(p)} q_j ≥ p 的最小索引集。你可以通过先按大小排序概率分布 𝑞，然后选择最大的词表元素直到达到目标水平 𝑝，来轻松计算这个量。

**问题（decoding）：解码（3 分）**

**交付物：** 实现一个从你的语言模型解码的函数。我们建议你支持以下特性：

- 为用户提供的提示词生成补全（即接收某个 𝑥_{1…𝑡} 并采样一个补全，直到遇到 `<|endoftext|>` token）。
- 允许用户控制最大生成 token 数。
- 给定期望的温度值，在采样之前对预测的下一个 token 分布应用 softmax 温度缩放。
- Top-𝑝 采样（[A. Holtzman et al., 2020] 也称为核采样），给定一个用户指定的阈值。

## 7 实验

现在是时候把所有东西组合起来，在预训练数据集上训练（小）语言模型了。

### 7.1 如何运行实验与交付物

理解 Transformer 架构组件背后原理的最好方式，是实际修改它并亲自运行。亲身体验无可替代。

为此，能够**快速、一致地做实验，并记录你做了什么**很重要。为了快速做实验，我们将在小规模模型（总共约 17M 参数）和简单数据集（TinyStories）上运行许多实验。为了保持一致，你将以系统化的方式消融（ablate）组件并改变超参数；为了记录，我们将要求你提交一份实验日志和与每个实验相关的学习曲线。

为了能够提交损失曲线，请确保周期性评估验证损失，并记录步数和挂钟时间。你可能会发现像 Weights and Biases 这样的日志基础设施很有帮助。

**问题（experiment_log）：实验日志（3 分）**

为你的训练和评估代码创建实验跟踪基础设施，使你能够跟踪你的实验，以及关于梯度步数和挂钟时间的损失曲线。

**交付物：** 用于实验的日志基础设施代码，以及一份实验日志（记录了你尝试过的所有事情），对应本节的下列作业问题。

### 7.2 TinyStories

我们将从一个非常简单的数据集（TinyStories；R. Eldan et al. [1]）开始，模型会在它上面快速训练，我们可以观察到一些有趣的行为。获取这个数据集的说明在第 1 节。下面这个数据集长什么样的例子。

**示例（tinystories_example）：TinyStories 中的一个例子**

> Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed! He said, "Wow, that is a really amazing vase! Can I buy it?" The shopkeeper smiled and said, "Of course you can. You can take it home and show all your friends how amazing it is!" So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was. And that's how Ben found an amazing vase in the store!
>
> （译文：从前有个叫 Ben 的小男孩。Ben 喜欢探索周围的世界。他看到了许多奇妙的东西，比如商店里陈列的漂亮花瓶。有一天，Ben 走过商店时，碰到一个非常特别的花瓶。Ben 看到它时惊叹不已！他说："哇，那真是个超棒的花瓶！我能买下它吗？"店主微笑着说："当然可以。你可以把它带回家，向所有朋友展示它有多棒！"于是 Ben 把花瓶带回了家，他为此非常自豪！他叫来朋友们，向他们展示了这个奇妙的花瓶。所有朋友都觉得这花瓶很漂亮，不敢相信 Ben 有多幸运。这就是 Ben 在商店里发现一个奇妙花瓶的故事！）

#### 7.2.1 超参数调优

我们会告诉你一些非常基本的超参数作为起点，并要求你自己找到其他一些运行良好的设置。

- **词表大小 10000。** 典型的词表大小在几万到几十万之间。你应该改变它，看看词表和模型行为如何变化。
- **上下文长度 256。** TinyStories 这样的简单数据集可能不需要很长的序列长度，但对于之后的 OpenWebText 数据，你可能想改变它。试着改变它，看看对每次迭代运行时间和最终困惑度的影响。
- **d_model 512。** 这比许多小型 Transformer 论文中使用的 768 维略小，但这样会更快。
- **d_ff 1344。** 这大约是 (8/3)·d_model，同时是 64 的倍数，对 GPU 性能有利。
- **RoPE theta 参数 Θ 10000。**
- **层数和头数：4 层、16 头。** 合起来，这大约给出 17M 个非嵌入参数，是一个相当小的 Transformer。
- **处理的总 token 数 327,680,000**（你的批量大小 × 总步数 × 上下文长度应大致等于这个值）。

你应该通过一些试错来为以下其他超参数找到好的默认值：学习率、学习率预热、其他 AdamW 超参数（𝛽1, 𝛽2, 𝜀）以及权重衰减。你可以在 D. P. Kingma et al. [22] 中找到这些超参数的一些典型选择。

#### 7.2.2 组合起来

现在你可以通过获得一个训练好的 BPE 分词器、分词训练数据集，并在你编写的训练循环中运行它，把所有东西组合起来。**重要提示：** 如果你的实现正确且高效，上述超参数在 1 块 B200 GPU 上应产生大约 20–30 分钟的运行时间。如果你的运行时间远长于此，请检查并确保你的数据加载、检查点或验证损失代码没有成为运行时间的瓶颈，并且你的实现已正确批处理。

#### 7.2.3 调试模型架构的技巧与窍门

我们强烈建议你熟悉 IDE 的内置调试器（例如 VSCode/Zed），这比用 print 语句调试更省时间。如果你使用文本编辑器，可以使用 `ipdb` 之类的东西。调试模型架构时的一些其他良好实践：

- 开发任何神经网络架构时，常见的第一步是**过拟合单个 minibatch**。如果你的实现正确，你应该能迅速把训练损失降到接近零。
- 在各种模型组件中设置调试断点，检查中间张量的形状，确保它们符合你的预期。
- 监控激活值、模型权重和梯度的范数，确保它们没有爆炸或消失。

**问题（learning_rate）：调学习率（2 B200 小时）（3 分）**

学习率是需要调优的最重要的超参数之一。基于你训练好的基础模型，回答以下问题：

(a) 对学习率做一次超参数搜索，并报告最终损失（如果优化器发散，则注明发散）。

**交付物：** 与多个学习率相关的学习曲线。解释你的超参数搜索策略。

**交付物：** 一个在 TinyStories 上验证损失（每 token）至多为 1.45 的模型。

> **低资源提示：在 CPU 或 Apple Silicon 上训练若干步**
>
> 如果你在 cpu 或 mps 上运行，你应该把处理的总 token 数减少到 40,000,000，这足以产生相当流畅的文本。你也可以把目标验证损失从 1.45 放宽到 2.00。
>
> 在配备 36 GB 内存的 M4 Max 芯片上用调优后的学习率运行我们的参考实现，我们使用批量大小 × 总步数 × 上下文长度 = 32×5000×256 = 40,960,000 token，在 cpu 上耗时 1 小时 22 分钟，在 mps 上耗时 36 分钟。在第 5000 步，我们达到 1.80 的验证损失。
>
> 一些额外提示：
> - 当使用 𝑁 个训练步时，我们建议调整余弦学习率衰减调度，使其恰好在第 𝑁 步终止衰减（即达到最小学习率）。
> - 使用 mps 时，不要使用 TF32 内核，即不要像使用 cuda 设备时那样设置 `torch.set_float32_matmul_precision('high')`。我们尝试在 mps 上启用 TF32 内核（torch 版本 2.9.0），发现后端有时会使用静默损坏的内核，导致训练不稳定。
> - 你可以通过用 `torch.compile` JIT 编译你的模型来加速训练。具体来说：
>   - 在 cpu 上，用 `model = torch.compile(model)` 编译你的模型。
>   - 在 mps 上，你可以用 `model = torch.compile(model, backend="aot_eager")` 在一定程度上优化反向传播。截至 torch 版本 2.9.0，mps 不支持用 Inductor 编译。

(b) 民间经验是，最佳学习率位于"稳定性边缘"。研究学习率发散的点与你的最佳学习率之间的关系。

**交付物：** 递增学习率的学习曲线，其中至少包含一次发散运行，以及对这与收敛速度之间关系的分析。

现在让我们改变批量大小，看看训练会发生什么。批量大小很重要——它们通过做更大的矩阵乘法让我们从 GPU 获得更高效率，但我们真的总是想要大的批量大小吗？让我们跑一些实验来找出答案。

**问题（batch_size_experiment）：批量大小变化（1 B200 小时）（1 分）**

把你的批量大小从 1 一路改变到 GPU 内存极限。中间至少尝试几个批量大小，包括 64 和 128 这样的典型大小。

**交付物：** 不同批量大小运行的学习曲线。如有必要，学习率应重新优化。

**交付物：** 几句话，讨论你关于批量大小及其对训练影响的发现。

有了手中的解码器，我们现在可以生成文本了！我们将从模型生成，看看它有多好。作为参考，你应该得到至少和下面例子一样好的输出。

**示例（ts_generate_example）：TinyStories 语言模型的样例输出**

> Once upon a time, there was a pretty girl named Lily. She loved to eat gum, especially the big black one. One day, Lily's mom asked her to help cook dinner. Lily was so excited! She loved to help her mom. Lily's mom made a big pot of soup for dinner. Lily was so happy and said, "Thank you, Mommy! I love you." She helped her mom pour the soup into a big bowl. After dinner, Lily's mom made some yummy soup. Lily loved it! She said, "Thank you, Mommy! This soup is so yummy!" Her mom smiled and said, "I'm glad you like it, Lily." They finished cooking and continued to cook together. The end.
>
> （译文：从前，有个叫 Lily 的漂亮女孩。她喜欢吃口香糖，尤其是那个大大的黑色口香糖。有一天，Lily 的妈妈让她帮忙做晚饭。Lily 非常兴奋！她喜欢帮妈妈。Lily 的妈妈做了一大锅汤当晚饭。Lily 很开心地说："谢谢你，妈妈！我爱你。"她帮妈妈把汤倒进一个大碗里。晚饭后，Lily 的妈妈做了一些美味的汤。Lily 很喜欢！她说："谢谢你，妈妈！这汤太好喝了！"她妈妈笑着说："你喜欢就好，Lily。"她们做完饭后继续一起做饭。完。）

> **低资源提示：在 CPU 或 Apple Silicon 上生成文本**
>
> 如果你使用的是处理 40M token 的低资源配置，你看到的生成结果应该仍然像英语，但不如上面的流畅。例如，我们在 40M token 上训练的 TinyStories 语言模型的样例输出如下：
>
> > Once upon a time, there was a little girl named Sue. Sue had a tooth that she loved very much. It was his best head. One day, Sue went for a walk and met a ladybug! They became good friends and played on the path together.
> >
> > "Hey, Polly! Let's go out!" said Tim. Sue looked at the sky and saw that it was difficult to find a way to dance shining. She smiled and agreed to help the talking!"
> >
> > As Sue watched the sky moved, what it was. She
>
> （译文：从前，有个叫 Sue 的小女孩。Sue 有一颗她非常喜爱的牙齿。那是他最好的头。有一天，Sue 去散步，遇到一只瓢虫！他们成了好朋友，一起在小路上玩耍。
>
> "嘿，Polly！我们出去玩吧！"Tim 说。Sue 看着天空，发现很难找到一种让舞蹈闪耀的方式。她微笑着同意帮助那个会说话的！"
>
> 当 Sue 看着天空移动时，它是什么。她）

这里是精确的问题陈述和我们要求的内容：

**问题（generate）：生成文本（1 分）**

使用你的解码器和训练好的检查点，报告你的模型生成的文本。你可能需要调整解码器参数（temperature、top-p 等）来获得流畅的输出。

**交付物：** 至少 256 个 token 的文本转储（或直到第一个 `<|endoftext|>` token），并简短评论该输出的流畅度，以及至少两个影响该输出好坏的因子。

### 7.3 消融与架构修改

理解 Transformer 的最好方式是实际修改它并观察它的行为。我们现在将做一些简单的消融和修改。

**消融 1：层归一化**

人们常说层归一化对 Transformer 训练的稳定性很重要。但也许我们想活得刺激一点。让我们从每个 Transformer 块中移除 RMSNorm，看看会发生什么。

**问题（layer_norm_ablation）：移除 RMSNorm 并训练（0.5 B200 小时）（1 分）**

从你的 Transformer 中移除所有 RMSNorm 并训练。在之前的最优学习率下会发生什么？你能通过使用更低的学习率获得稳定性吗？

**交付物：** 当你移除 RMSNorm 并训练时的一条学习曲线，以及最佳学习率的一条学习曲线。

**交付物：** 几句话评论 RMSNorm 的影响。

现在让我们研究另一种乍看之下很随意的层归一化选择。Pre-norm Transformer 块定义为

𝑧 = 𝑥 + MultiHeadSelfAttention(RMSNorm(𝑥))    (25)

𝑦 = 𝑧 + FFN(RMSNorm(𝑧)).    (26)

这是对原始 Transformer 架构少有的几项"共识性"修改之一，原始架构使用 post-norm 方法：

𝑧 = RMSNorm(𝑥 + MultiHeadSelfAttention(𝑥))    (27)

𝑦 = RMSNorm(𝑧 + FFN(𝑧)).    (28)

让我们回到 post-norm 方法，看看会发生什么。

**问题（pre_norm_ablation）：实现 post-norm 并训练（0.5 B200 小时）（1 分）**

把你的 pre-norm Transformer 实现修改成 post-norm 的。用 post-norm 模型训练，看看会发生什么。

**交付物：** post-norm Transformer 的一条学习曲线，与 pre-norm 的进行比较。

我们看到层归一化对 Transformer 的行为有重大影响，甚至层归一化的**位置**也很重要。

**消融 2：位置嵌入**

接下来我们将研究位置嵌入对模型性能的影响。具体来说，我们将比较我们的基础模型（带 RoPE）与完全不包含位置嵌入（NoPE）的情况。事实证明，decoder-only Transformer（即像我们实现的这样带因果掩码的）在理论上可以在不被显式提供位置嵌入的情况下推断出相对或绝对位置信息 [Y.-H. H. Tsai et al., 2019; A. Kazemnejad et al., 2023]。我们现在将经验性地测试 NoPE 与 RoPE 相比表现如何。

**问题（no_pos_emb）：实现 NoPE（0.5 B200 小时）（1 分）**

修改你带 RoPE 的 Transformer 实现，完全移除位置嵌入信息，看看会发生什么。

**交付物：** 比较 RoPE 与 NoPE 性能的一条学习曲线。

**消融 3：SwiGLU vs. SiLU**

接下来，我们将遵循 N. Shazeer [20]，测试前馈网络中门控的重要性，比较 SwiGLU 前馈网络与使用 SiLU 激活但无门控线性单元（GLU）的前馈网络的性能：

FFN_SiLU(𝑥) = 𝑊2 · SiLU(𝑊1𝑥).    (29)

回顾一下，在我们的 SwiGLU 实现中，我们把内部前馈层的维度设为约 d_ff = (8/3)·d_model（同时确保 d_ff mod 64 = 0，以利用 GPU tensor core）。在这个消融基线中，你的 FFN_SiLU 实现应改为设 d_ff = 4×d_model，以大致匹配默认 SwiGLU 前馈网络（它有三个而非两个权重矩阵）的参数数量。

**问题（swiglu_ablation）：SwiGLU vs. SiLU（0.5 B200 小时）（1 分）**

**交付物：** 比较 SwiGLU 与 SiLU 前馈网络性能的一条学习曲线，参数数量大致匹配。

**交付物：** 几句话讨论你的发现。

> **低资源提示：GPU 资源有限的在线学生应在 TinyStories 上测试修改**
>
> 在作业的其余部分，我们将转向更大规模、噪声更多的网络数据集（OpenWebText），实验架构修改，并（可选地）向课程排行榜提交。
>
> 在 OpenWebText 上把 LM 训练到流畅需要很长时间，所以我们建议 GPU 访问受限的在线学生继续在 TinyStories 上测试修改（使用验证损失作为评估性能的指标）。

### 7.4 在 OpenWebText 上运行

我们现在转向一个更标准的、从网络爬虫创建的预训练数据集。OpenWebText [A. Gokaslan et al., 2019] 的一小部分样本也作为单个文本文件提供：如何访问该文件见第 1 节。

这是 OpenWebText 中的一个例子。注意，文本要真实、复杂和多样得多。你可能想浏览一下训练数据集，感受一下网络爬取语料的训练数据是什么样的。

**示例（owt_example）：OWT 中的一个例子**

> Baseball Prospectus director of technology Harry Pavlidis took a risk when he hired Jonathan Judge.
>
> Pavlidis knew that, as Alan Schwarz wrote in The Numbers Game, "no corner of American culture is more precisely counted, more passionately quantified, than performances of baseball players." With a few clicks here and there, you can find out that Noah Syndergaard's fastball revolves more than 2,100 times per minute on its way to the plate, that Nelson Cruz had the game's highest average exit velocity among qualified hitters in 2016 and myriad other tidbits that seem ripped from a video game or science fiction novel. The rising ocean of data has empowered an increasingly important actor in baseball's culture: the analytical hobbyist.
>
> That empowerment comes with added scrutiny – on the measurements, but also on the people and publications behind them. With Baseball Prospectus, Pavlidis knew all about the backlash that accompanies quantitative imperfection. He also knew the site's catching metrics needed to be reworked, and that it would take a learned mind – someone who could tackle complex statistical modeling problems – to complete the job.
>
> "He freaks us out." Harry Pavlidis
>
> Pavlidis had a hunch that Judge "got it" based on the latter's writing and their interaction at a site-sponsored ballpark event. […]
>
> （译文：Baseball Prospectus 的技术总监 Harry Pavlidis 在雇用 Jonathan Judge 时冒了风险。
>
> Pavlidis 知道，正如 Alan Schwarz 在《The Numbers Game》中所写："美国文化中没有任何一个角落比棒球运动员的表现被更精确地计数、更热烈地量化。"点几下鼠标，你就能发现 Noah Syndergaard 的快速球在飞向本垒板的途中每分钟旋转超过 2,100 次，Nelson Cruz 在 2016 年拥有合格打者中最高的平均出球速度，以及无数其他仿佛出自电子游戏或科幻小说的小知识。不断上升的数据海洋赋予棒球文化中一个日益重要的角色以力量：分析型爱好者。
>
> 这种赋权伴随着更多的审视——不仅是对测量结果的审视，也是对背后的人和出版物的审视。在 Baseball Prospectus 的经历让 Pavlidis 深知量化不完美所伴随的反弹。他也知道该网站的接球指标需要重做，而这需要一位博学之人——一个能处理复杂统计建模问题的人——来完成这项工作。
>
> "他让我们害怕。"Harry Pavlidis
>
> Pavlidis 凭直觉认为 Judge"懂了"，这基于后者的写作以及他们在一场网站赞助的棒球场活动中的互动。[…]）

注意：对于这个实验，你可能需要重新调优你的超参数，如学习率或批量大小。

**问题（main_experiment）：在 OWT 上实验（2 B200 小时）（2 分）**

用与 TinyStories 相同的模型架构和总训练迭代数，在 OpenWebText 上训练你的语言模型。这个模型表现如何？

**交付物：** 你的语言模型在 OpenWebText 上的一条学习曲线。描述与 TinyStories 的损失差异——我们应该如何解读这些损失？

**交付物：** 从 OpenWebText LM 生成的文本，格式与 TinyStories 输出相同。该文本的流畅度如何？为什么即使我们有与 TinyStories 相同的模型和计算预算，输出质量却更差？

### 7.5 你自己的修改 + 排行榜

恭喜你走到这一步。你几乎完成了！现在你将尝试改进 Transformer 架构，看看你的超参数和架构与班上其他同学相比如何。

**排行榜规则**

除了以下限制，没有任何限制：

- **运行时间：** 你的提交在 B200 上最多运行 45 分钟。如果你使用 SLURM 或 Modal，可能需要在提交脚本中强制执行这一点。
- **数据：** 你只能使用我们提供的 OpenWebText 训练数据集。

除此之外，你可以随心所欲地做任何事。

如果你在寻找实现什么的想法，可以看看这些资源：

- 最先进的开源 LLM 家族，如 Llama 3 [A. Grattafiori et al., 2024] 或 Qwen 2.5 [A. Yang et al., 2024]。
- NanoGPT speedrun 仓库（github.com/KellerJordan/modded-nanogpt），社区成员在那里发布了许多有趣的修改，用于"速通"小规模语言模型预训练。例如，一个可追溯到原始 Transformer 论文的常见修改是**把输入和输出嵌入的权重绑定（tie）在一起**（见 A. Vaswani et al. [8]（第 3.4 节）和 A. Chowdhery et al. [16]（第 2 节））。如果你确实尝试权重绑定，可能必须降低 embedding/LM head 初始化的标准差。

在尝试完整的 45 分钟运行之前，你会想在 OpenWebText 的一个小子集上或 TinyStories 上测试这些。

作为一个提醒，我们确实指出，你在这个排行榜上可能发现有效的一些修改可能不会泛化到更大规模的预训练。我们将在课程的缩放定律单元中进一步探讨这个想法。

**问题（leaderboard）：排行榜（10 B200 小时）（6 分）**

你将在上述排行榜规则下训练一个模型，目标是在 0.75 B200 小时内最小化你的语言模型的验证损失。

**交付物：** 记录的最终验证损失、一条清晰显示挂钟时间 x 轴小于 45 分钟的关联学习曲线，以及对你所做事情的描述。我们期望排行榜提交至少能击败 5.0 损失的朴素基线。在此提交到排行榜：github.com/stanford-cs336/assignment1-basics-leaderboard。

## 参考文献

[1] R. Eldan and Y. Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?." 2023.

[2] A. Gokaslan, V. Cohen, E. Pavlick, and S. Tellex, "OpenWebText corpus." 2019.

[3] R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," in Proc. of ACL, 2016.

[4] C. Wang, K. Cho, and J. Gu, "Neural Machine Translation with Byte-Level Subwords." 2019.

[5] P. Gage, "A new algorithm for data compression," C Users Journal, vol. 12, no. 2, pp. 23–38, Feb. 1994.

[6] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language Models are Unsupervised Multitask Learners." 2019.

[7] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, "Improving Language Understanding by Generative Pre-Training." 2018.

[8] A. Vaswani et al., "Attention is All you Need," in Proc. of NeurIPS, 2017.

[9] T. Q. Nguyen and J. Salazar, "Transformers without Tears: Improving the Normalization of Self-Attention," in Proc. of IWSWLT, 2019.

[10] R. Xiong et al., "On Layer Normalization in the Transformer Architecture," in Proc. of ICML, 2020.

[11] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer Normalization." 2016.

[12] H. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models." 2023.

[13] B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," in Proc. of NeurIPS, 2019.

[14] A. Grattafiori et al., "The Llama 3 Herd of Models." [Online]. Available: https://arxiv.org/abs/2407.21783

[15] A. Yang et al., "Qwen2.5 Technical Report," arXiv preprint arXiv:2412.15115, 2024.

[16] A. Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways." 2022.

[17] D. Hendrycks and K. Gimpel, "Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units." 2016.

[18] S. Elfwing, E. Uchibe, and K. Doya, "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning." [Online]. Available: https://arxiv.org/abs/1702.03118

[19] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, "Language Modeling with Gated Convolutional Networks." [Online]. Available: https://arxiv.org/abs/1612.08083

[20] N. Shazeer, "GLU Variants Improve Transformer." 2020.

[21] J. Su, Y. Lu, S. Pan, B. Wen, and Y. Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.

[22] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in Proc. of ICLR, 2015.

[23] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in Proc. of ICLR, 2019.

[24] T. B. Brown et al., "Language Models are Few-Shot Learners," in Proc. of NeurIPS, 2020.

[25] J. Kaplan et al., "Scaling Laws for Neural Language Models." 2020.

[26] J. Hoffmann et al., "Training Compute-Optimal Large Language Models." 2022.

[27] A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, "The Curious Case of Neural Text Degeneration," in Proc. of ICLR, 2020.

[28] Y.-H. H. Tsai, S. Bai, M. Yamada, L.-P. Morency, and R. Salakhutdinov, "Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), K. Inui, J. Jiang, V. Ng, and X. Wan, Eds., Hong Kong, China: Association for Computational Linguistics, Nov. 2019, pp. 4344–4353. doi: 10.18653/v1/D19-1443.

[29] A. Kazemnejad, I. Padhi, K. Natesan, P. Das, and S. Reddy, "The Impact of Positional Encoding on Length Generalization in Transformers," in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview.net/forum?id=Drrl2gcjzl

