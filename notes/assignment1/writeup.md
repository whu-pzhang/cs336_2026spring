# CS336 Assignment 1 Writeup（草稿）

> 作业1（Basics）书面题草稿。每题先附题干，再写回答。最终提交需排版为 `writeup.pdf`（课程通常用英文；本稿先用中文）。
>
> 其他作业的笔记见 `notes/assignment2/` 等目录。

## 2 Byte-Pair Encoding (BPE) Tokenizer

### unicode1：Understanding Unicode（1 分）

#### 题干

**(a)** What Unicode character does `chr(0)` return?  
Deliverable: A one-sentence response.

**(b)** How does this character’s string representation (`__repr__()`) differ from its printed representation?  
Deliverable: A one-sentence response.

**(c)** What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter:

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

Deliverable: A one-sentence response.

#### 回答

**(a)** `chr(0)` 返回 Unicode 空字符（null character，码点 U+0000），在 Python
中通常用转义形式 `\x00` 表示。

**(b)** `repr(chr(0))` 会显示可见的转义字符串 `'\x00'`，而 `print(chr(0))`
会直接写出 NUL 控制字符，终端通常不显示任何可见符号。

**(c)** NUL 不会终止 Python 字符串：拼接后字符串仍包含它且 `len` 会计数，但
`print("this is a test" + chr(0) + "string")` 中间的 NUL 通常不可见，所以视觉上
可能像是两段文字直接相连。

---



### unicode2：Unicode Encodings（3 分）



#### 题干

**(a)** What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.  
Deliverable: A one-to-two sentence response.

**(b)** Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

Deliverable: An example input byte string for which `decode_utf8_bytes_to_str_wrong` produces incorrect output, with a one-sentence explanation of why the function is incorrect.

**(c)** Give a two-byte sequence that does not decode to any Unicode character(s).  
Deliverable: An example, with a one-sentence explanation.

#### 回答

**(a)** UTF-8 对 ASCII 字符只需 1 个字节，而 UTF-16 和 UTF-32 通常分别需要
2 和 4 个字节；例如 `"hello! こんにちは!"` 的编码长度分别是 UTF-8 **23**、
UTF-16-LE **26**、UTF-32-LE **52** bytes（不计 BOM）。UTF-8 还是互联网的主流编码，并且只需固定的
256 个单字节基本符号即可覆盖所有 Unicode 输入。

**(b)** 示例输入：`"牛".encode("utf-8")`（即 `b'\xe7\x89\x9b'`）；该函数把一个
多字节 UTF-8 序列拆成三个单字节分别解码，首字节和续字节都不是独立的合法字符，
因此会抛出 `UnicodeDecodeError` 而不是返回 `"牛"`。

**(c)** 示例：`b'\xc0\x80'`。这是非法（overlong）的 UTF-8 两字节序列，调用 `.decode("utf-8")` 会抛出 `UnicodeDecodeError`。

---



### train_bpe_tinystories / train_bpe_expts_owt：BPE Training（4 分）

#### 题干

**TinyStories (a)** Train a byte-level BPE tokenizer on TinyStories with a
maximum vocabulary size of 10,000, add the `<|endoftext|>` special token, and
serialize the vocabulary and merges. Report the training time and memory, and
inspect the longest vocabulary token.

**TinyStories (b)** Profile the code and identify which part of tokenizer
training takes the most time.

**OpenWebText (a)** Train a byte-level BPE tokenizer on OpenWebText with a
maximum vocabulary size of 32,000, serialize the vocabulary and merges, and
inspect the longest vocabulary token.

**OpenWebText (b)** Compare and contrast the tokenizers trained on TinyStories
and OpenWebText.

#### 回答

本次可复现实验使用仓库中的 `tinystories_50M.txt`（50M 字符）和
`owt_5M.txt`（5M 字符）子集；完整配置、词表、merges 和训练元数据分别保存在
`assignment1-basics/experiments/artifacts/tinystories_10k/` 与
`assignment1-basics/experiments/artifacts/owt_32k/`。

**TinyStories (a)** 使用 10,000 词表（包含 `<|endoftext|>`，以及数据管线使用的
`<|pad|>`）训练耗时约 **31.5 秒**；最长 token 是长度 **17** 的
`" enthusiastically"`，是一个带前导空格的完整英文单词，符合 TinyStories
中高频、简单英文叙事文本的分布。训练过程没有单独采集峰值 RSS，但远低于题目
要求的 30GB 内存上限。

**TinyStories (b)** 优化后的训练流程中，预分词只需遍历语料一次；主要耗时来自
BPE merge 循环中维护 pair 频数、反向索引并不断更新受影响的 pre-token。使用
`pair_to_words` 和最大频数堆后，每轮只处理包含当前 pair 的词，避免重新扫描整个
词表，因而显著降低了训练时间。实测预分词约耗时 6.2 秒（50M 子集），而总训练
耗时为 31.5 秒，因此 merge 迭代是主要瓶颈。

**OpenWebText (a)** 使用 32,000 词表训练耗时约 **389.4 秒（6.5 分钟）**；最长
token 长度为 **51**，内容是一个重复出现的长数字串
`100000000000000088817841970012523233890533447265625`。它不是自然语言词，
但在网页语料中某些数字串会高频重复，因此被 BPE 合并成较长 token 是合理的。

**OpenWebText (b)** TinyStories tokenizer 更偏向儿童故事中的常见英文词和短句；
OpenWebText tokenizer 的词表更大，覆盖网页中的数字、代码、URL、多语言和各种
格式，因此能学习到更多长而专门化的 token。相应地，两个 tokenizer 的 merge
规则取决于训练语料分布，不能简单认为较大的词表在所有领域都更优。

---

### tokenizer_experiments：Experiments with tokenizers（4 分）

#### 题干

**(a)** Sample 10 documents from TinyStories and OpenWebText. Using the
previously-trained TinyStories (10K) and OpenWebText (32K) tokenizers, encode
the sampled documents into integer IDs. What is each tokenizer's compression
ratio (bytes/token)?

**(b)** What happens if you tokenize the OpenWebText sample with the TinyStories
tokenizer? Compare the compression ratio and/or qualitatively describe what
happens.

**(c)** Estimate the throughput of your tokenizer (e.g., in bytes/second). How
long would it take to tokenize the Pile dataset (825GB of text)?

**(d)** Using your TinyStories and OpenWebText tokenizers, encode the respective
training and development datasets into a sequence of integer token IDs.

#### 回答

实验使用随机种子 `42`，每个数据集抽取 10 个文档；结果保存于
`assignment1-basics/experiments/artifacts/tokenizer_experiments.json`。

**(a)** `bytes/token` 越大表示压缩效果越好。TinyStories 样本使用 TinyStories
tokenizer 时为 **4.044 bytes/token**，使用 OWT tokenizer 时为 **3.857
bytes/token**。OWT 样本使用 TinyStories tokenizer 时为 **3.588 bytes/token**，
使用 OWT tokenizer 时为 **4.323 bytes/token**。

**(b)** 在 OWT 样本上，TinyStories tokenizer 的压缩率从 OWT tokenizer 的
4.323 降到 3.588 bytes/token，即需要更多 token 表示相同字节数，压缩效果约差
17%。这符合 tokenizer 对训练语料分布有适应性：TinyStories 词表更偏向短小、
简单的儿童故事文本，而 OWT 包含更广泛的网页词汇、数字和格式。

**(c)** 将四组样本的测量合并后，TinyStories tokenizer 吞吐量约为 **3.68
MB/s**，处理 825GB Pile 预计需要 **62.25 小时（约 2.59 天）**；OWT
tokenizer 吞吐量约为 **3.04 MB/s**，预计需要 **75.33 小时（约 3.14 天）**。
该估算使用 `825,000,000,000 / throughput`，实际时间会随硬件、进程数和输入
缓存情况变化。

**(d)** `tokenize_datasets.py` 已实现按 special-token 边界分块，并用 8 个 worker
通过两个 tokenizer 的 `encode_iterable` 生成 `uint16` 的 `.npy` token-ID 序列，
供后续训练循环通过 memory map 加载。当前目录中的数组是此前 32,768 词表配置
生成的；要严格使用本题要求的 TinyStories 10K / OWT 32K artifact，需要在
`assignment1-basics` 目录运行下面的命令重新生成（不要使用
`--skip-existing`）：

```bash
./.venv/bin/python experiments/tokenize_datasets.py \
  --dataset both --split both --workers 8
```

---

## 3 Transformer Language Model

> 实现题见 `assignment1-basics/` 代码。此处只收录书面题。题干以课程讲义为准；若与本地 `cs336_assignment1_basics.pdf` 有出入，以 PDF 为准。



### transformer_accounting：Transformer LM resource accounting（5 分）



#### 题干

大部分 Transformer 的浮点运算来自矩阵乘。对 $A \in \mathbb{R}^{m \times n}$、$B \in \mathbb{R}^{n \times p}$，乘积 $AB$ 计 **$2mnp$ FLOPs**。下面的模型指 **本作业实现的架构**（SwiGLU、RMSNorm、RoPE、无 bias），不是 HuggingFace 的 GPT-2 原版。

**(a)** Consider GPT-2 XL, which has the following configuration:

- `vocab_size`: 50,257
- `context_length`: 1,024
- `num_layers`: 48
- `d_model`: 1,600
- `num_heads`: 25
- `d_ff`: 4,288

Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

Deliverable: The number of trainable parameters, and the memory required to load the model.

**(b)** Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has `context_length` tokens.

Deliverable: A list of the matrix multiplies, and the total number of FLOPs.

**(c)** Based on your analysis above, which parts of the model require the most FLOPs?

Deliverable: A one-to-two sentence response.

**(d)** Repeat your analysis with GPT-2 small (12 layers, 768 `d_model`, 12 heads), GPT-2 medium (24 layers, 1024 `d_model`, 16 heads), and GPT-2 large (36 layers, 1280 `d_model`, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

For each model, provide a breakdown of model components and its associated FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.

Deliverable: A table (or equivalent) of FLOP proportions by component for each model size, plus a short description of the trend.

**(e)** Take the GPT-2 XL-shaped model and increase `context_length` to 16,384. How do the FLOPs of the various components change? Which parts of the model take up proportionally more or less of the total FLOPs?

Deliverable: A description of how component FLOPs (and their proportions) change at the longer context length.

#### 回答

**(a)**

参数量由 token embedding、每层的 MHA/FFN/两个 RMSNorm、最终 RMSNorm 和
lm head 组成。由于 RoPE 没有可训练参数、Linear 没有 bias，

$$
\begin{aligned}
P &= 2(V D) + L(4D^2 + 3D d_{ff} + 2D) + D \\
  &= 2(50257)(1600) + 48(4\cdot1600^2 + 3\cdot1600\cdot4288 + 2\cdot1600) + 1600 \\
  &= 1,640,452,800 \text{ parameters}.
\end{aligned}
$$

float32 每个参数占 4 bytes，因此仅加载模型参数需要
**6,561,811,200 bytes（约 6.56 GB，6.11 GiB）**。

**(b)**

设输入序列长度为 `T=1024`。每层 MHA 的矩阵乘包括 Q/K/V 三个投影、
`QKᵀ`、注意力加权求和以及输出投影：

$$
F_{MHA,layer}=8TD^2+4T^2D=27.6824\text{ GFLOPs}.
$$

SwiGLU 的三个线性层需要

$$
F_{FFN,layer}=6TDd_{ff}=42.1528\text{ GFLOPs}.
$$

因此 48 层 block 共需 `48 × (27.6824 + 42.1528) = 3352.0878 GFLOPs`。
最后的 lm head 需要
`2TDV = 164.6821 GFLOPs`，总前向计算量为
**3516.7699 GFLOPs（约 3.5168 TFLOPs）**。

**(c)** FFN 层占了最多的 FLOPs，约为 **57.53%**；MHA 约占 37.78%，lm head
约占 4.68%。

**(d)** 

在 `context_length=1024`、`vocab_size=50257` 下，各模型的矩阵乘 FLOPs 如下；
MHA 已包含 QKV/output 投影和两次 attention 矩阵乘。

| Model       | num_layers | d_model | num_heads | d_ff | MHA FLOPs | FFN FLOPs | lm_head FLOPs |
| ----------- | ---------- | ------- | --------- | ---- | --------- | --------- | ------------- |
| GPT2-small  | 12         | 768     | 12        | 2048 | 96.6368   | 115.9641  | 79.0474       |
| GPT2-medium | 24         | 1024    | 16        | 2752 | 309.2376  | 415.5381  | 105.3966      |
| GPT2-large  | 36         | 1280    | 20        | 3392 | 676.4573  | 960.3278  | 131.7457      |
| GPT2-XL     | 48         | 1600    | 25        | 4288 | 1328.7555 | 2023.3322 | 164.6821      |


各模块占比如下：
| Model       | Total FLOPs | MHA   | FFN   | lm_head |
| ----------- | ----------- | ----- | ----- | ------- |
| GPT2-small  | 291.6483    | 33.13% | 39.76% | 27.10% |
| GPT2-medium | 830.1723    | 37.25% | 50.05% | 12.70% |
| GPT2-large  | 1768.5309   | 38.25% | 54.30% | 7.45%  |
| GPT2-XL     | 3516.7699   | 37.78% | 57.53% | 4.68%  |


随着模型参数增加，不同模块FLOPs趋势如下：

- FFN：约 39.76% → 57.53%，随着 `d_model` 和层数增加，份额持续上升。
- lm_head：约 27.10% → 4.68%，因为词表大小固定，其比例下降最明显。
- MHA：约 33.13% → 37.78%，比例略升后趋于稳定。

**(e)**

将 GPT-2 XL 的 `T` 从 1024 增大到 16,384（16 倍）时，逐 token 的投影、FFN
和 lm head FLOPs 都乘以 16，而 `QKᵀ` 与 attention 加权求和乘以 `16²=256`。
具体结果如下：

| Component | FLOPs at `T=16,384` | Share |
|---|---:|---:|
| MHA | 98.5695 TFLOPs | 73.79% |
| FFN | 32.3733 TFLOPs | 24.24% |
| lm_head | 2.6349 TFLOPs | 1.97% |
| Total | 133.5777 TFLOPs | 100% |

因此长上下文下 attention 的二次复杂度成为主要成本；FFN 和 lm head 虽然也增加
16 倍，但在总 FLOPs 中的比例明显下降。

---

## 4 Training a Transformer LM

### learning_rate_tuning：Tuning the learning rate（1 分）

#### 题干

Run the toy SGD example with learning rates `1e1`, `1e2`, and `1e3` for 10
training iterations. Compare the loss behavior with the baseline learning rate
`1`: does it decay faster, decay slower, or diverge?

#### 回答

实验脚本为 `assignment1-basics/experiments/learning_rate_tuning.py`，使用固定随机
种子 `0`、相同初始权重和 10 次更新；完整 loss 序列保存于
`assignment1-basics/experiments/artifacts/learning_rate_tuning.json`。

| learning rate | iteration 1 loss | iteration 10 loss | 行为 |
|---:|---:|---:|---|
| `1` | 26.2714 | 21.7399 | 缓慢下降 |
| `10` | 26.2714 | 3.53075 | 快速下降 |
| `100` | 26.2714 | 2.3499×10⁻²³ | 先振荡后快速下降 |
| `1000` | 26.2714 | 2.43501×10¹⁸ | 发散 |

增大学习率在稳定范围内可以加快收敛，但超过稳定范围后会导致更新过大、loss
迅速增长。

---

### adamw_accounting：Resource accounting for training with AdamW（2 分）

#### 题干

Assume we are using float32 for every tensor.

**(a)** How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of the `batch_size` and the model hyperparameters (`vocab_size`, `context_length`, `num_layers`, `d_model`, `num_heads`). Assume `d_ff = (8/3) × d_model`.

For simplicity, when calculating memory usage of activations, consider only the components listed in the handout (Transformer block internals, final RMSNorm, output embedding, cross-entropy on logits).

Deliverable: A peak-memory expression decomposed into parameters / activations / gradients / optimizer state.

**(b)** Instantiate your answer for a GPT-2 XL-shaped model (with **this assignment’s architecture**, not the original HuggingFace GPT-2) to get an expression that only depends on `batch_size`. What is the maximum batch size you can use and still fit within 80GB memory?

Deliverable: An expression of the form \(a \cdot \text{batch\_size} + b\), and the maximum batch size.

**(c)** How many FLOPs does running one step of AdamW take?

Deliverable: A FLOP count (or a tight estimate) with a brief justification.

**(d)** Model FLOPs utilization (MFU) is the ratio of observed throughput (in FLOP/s) to the hardware’s theoretical peak FLOP throughput. An NVIDIA H100 has a theoretical peak of 495 teraFLOP/s for float32 (TF32). Assuming 50% MFU, how long would it take to train a GPT-2 XL-shaped model for 400K steps with batch size 1024 on a single H100? Assume the backward pass has twice the FLOPs of the forward pass.

Deliverable: The number of hours, with a brief justification.

> 题干以 `cs336_assignment1_basics.pdf` 为准；若 PDF 里激活列表或 `d_ff` 假设与上文不完全一致，以 PDF 为准。

#### 回答

令 `b = batch_size`、`V = vocab_size`、`T = context_length`、
`L = num_layers`、`D = d_model`、`H = num_heads`，并按题目取
`d_ff = (8/3)D`。定义参数总数（以 float32 元素计）为

$$
P = 2VD + L(4D^2 + 3Dd_{ff} + 2D) + D
  = 2VD + L(12D^2 + 2D) + D.
$$

**(a)** 参数占用 `4P` bytes，梯度占用 `4P` bytes，AdamW 的一阶和二阶矩占用
`8P` bytes。按题目列出的中间张量计数，Transformer block 的激活元素数为
`(8bTD + 2bHT² + 4bTd_ff + bTD)`，因此总激活元素数为

$$
A = L\left(\frac{56}{3}bTD + 2bHT^2\right) + bTD + bTV.
$$

所以峰值显存表达式为

$$
\text{Memory}_{\text{peak}} = 16P + 4A \quad \text{bytes}.
$$

这里的 `16P` 来自参数、梯度和两个 AdamW 状态张量；激活项 `4A` 随 batch size
线性增长。

**(b)** 对 GPT-2 XL-shaped 配置 `V=50257`、`T=1024`、`L=48`、`D=1600`、
`H=25`，得到

$$
P = 1,635,537,600,
$$

以及

$$
\text{Memory}_{\text{peak}}(b)
 = 16,150,761,472b + 26,168,601,600\ \text{bytes}
 = 15.0416b + 24.3714\ \text{GiB}.
$$

在 80GB（无论按十进制 GB 或二进制 GiB 近似）限制下，最大整数 batch size
为 **3**；`b=4` 时约需 84.5 GiB，超出限制。

**(c)** 设参数量为 `P`。忽略少量标量 bias-correction 运算，每个参数的一次
AdamW 更新约包含：权重衰减 2 FLOPs、一阶矩 3 FLOPs、二阶矩 4 FLOPs，以及
归一化更新 5 FLOPs。因此

$$
\text{FLOPs}_{\text{AdamW step}} \approx 14P
 = 22,897,526,400 \approx 22.9\ \text{GFLOPs}.
$$

不同 fused kernel 对加法、乘法和开方的计数约定可能略有差异，但该数量级远小于
一次 GPT-2 XL 前向和反向传播的计算量。

**(d)** GPT-2 XL 在 `T=1024` 时单条序列前向约需 `3.5168 TFLOPs`。按题目假设，
一次训练 step（前向 + 反向）约为 `3 × 3.5168 TFLOPs`，因此 400K steps、
batch size 1024 的总计算量为

$$
3 \times 3.5168\times10^{12}\times1024\times400000
\approx 4.3214\times10^{21}\ \text{FLOPs}.
$$

H100 在 50% MFU 下的有效吞吐量为 `0.5 × 495 TFLOPs/s = 247.5 TFLOPs/s`，
所以预计训练时间为约 **4,850 小时（约 202 天）**。该估算只计算模型前向和反向，
没有额外计入数据读取、评估、checkpoint 和 AdamW 更新的开销。
