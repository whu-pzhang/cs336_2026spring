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

**(a)** `chr(0)` 返回的是 Unicode 空字符（null character，U+0000）；`\x00` 是它在 Python 中的转义写法。

**(b)** `repr`（或在 REPL 中直接查看）会显示转义形式 `'\x00'`，便于看见该字符；`print` 则输出字符本身，在终端上通常不可见。

**(c)** 字符串里确实包含该 null 字符，`len` 会把它算进去；但用 `print` 打印时，终端常把它当作控制字符处理，因此显示结果可能与 `repr` / 长度不一致。

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

**(a)** 更倾向用 UTF-8 字节训练 tokenizer：对 ASCII 只需 1 个字节，而 UTF-16 / UTF-32 至少要 2 / 4 个字节，对英文为主的语料更浪费空间；同时 UTF-8 也是互联网上最主流的编码。

**(b)** 示例输入：`"牛".encode("utf-8")`（即 `b'\xe7\x89\x9b'`）。该函数把每个字节单独 `decode("utf-8")`，无法正确处理多字节 UTF-8 字符（如汉字），续字节本身不是合法的单字节 UTF-8 序列。

**(c)** 示例：`b'\xc0\x80'`。这是非法（overlong）的 UTF-8 两字节序列，调用 `.decode("utf-8")` 会抛出 `UnicodeDecodeError`。

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

参数量：

1. embedding `vocab_size*d_model`
2. MHA: `qkv+output_proj: d_model*d_model*4`
3. FFN: `d_model*d_ff*3`
4. RMSNorm: `d_model`
5. ln_final: `d_model`
6. lm_head `vocab_size*d_model`

总参数量=`embedding+num_layers*(MHA+FFN+2*RMSNorm)+ln_final+lm_head`
       =80411200+48*(10240000+20582400+3200)+1600+80411200
       =1640452800=1.64B

若采用单精度，需占用显存 6.56 GB

**(b)** 

FLOPS：

1. MHA:
  QKV投影:`3*(2*T*d_model*d_model)=3*2*1024*1600^2=15.73 GFLOPs`
    QK^t: `2*T^2*d_model=2*1024^2*1600=3.35 GFLOPs`
    attn*V: `2*T^2*d_model=3.35 GFLOPs`
    output投影: `2*T*d_model^2=2*1024*1600^2=5.24 GFLOPs`
    单个MHA总计 15.73+3.35*2+5.24=27.67 GFLOPs
2. FFN:
  W1: `2*T*d_model*d_ff=2*1024*1600*4288=14.05 GFLOPs`
    W2: `2*T*d_ff*d_model=2*1024*4288*1600=14.05 GFLOPs`
    W3: `2*T*d_model*d_ff=2*1024*1600*4288=14.05 GFLOPs`
    单个FFN总计 14.05*3=42.15 GFLOPs

一个TransformerBlock总计 27.67+42.15=69.82 GFLOPs
48层总计 69.82*48=3351.36 GFLOPs

1. lm_head: `2*T*d_model*vocab_size=2*1024*1600*50257=164.68 GFLOPs`

总共 FLOPs=3351.36+164.68=3,516.04 GFLOPs=3.52 TFLOPs

**(c)** FFN层占了最多的FLOPs，占比约 57.5%

**(d)** 

context_length=1024时，各模型MHA和FFN FLOPs如下：

| Model       | num_layers | d_model | num_heads | d_ff | MHA FLOPs | FFN FLOPs | lm_head FLOPs |
| ----------- | ---------- | ------- | --------- | ---- | --------- | --------- | ------------- |
| GPT2-small  | 12         | 768     | 12        | 2048 | 96.6      | 115.92    | 79.05         |
| GPT2-medium | 24         | 1024    | 16        | 2752 | 309.36    | 415.44    | 105.4         |
| GPT2-large  | 36         | 1280    | 20        | 3392 | 676.44    | 960.48    | 131.75        |
| GPT2-XL     | 48         | 1600    | 25        | 4288 | 1328.16   | 2023.2    | 164.68        |


各模块占比如下：
| Model       | Total FLOPs | MHA   | FFN   | lm_head |
| ----------- | ----------- | ----- | ----- | ------- |
| GPT2-small  | 291.57      | 33.1% | 39.8% | 27.1%   |
| GPT2-medium | 830.2       | 37.3% | 50.0% | 12.7%   |
| GPT2-large  | 1768.67     | 38.2% | 54.3% | 7.4%    |
| GPT2-XL     | 3516.04     | 37.8% | 57.5% | 4.7%    |


随着模型参数增加，不同模块FLOPs趋势如下：

- FFN：约 40% → 58%，份额上升
- lm_head：约 27% → 5%，下降最明显（vocab 固定）
- MHA：约 33% → 38%，先升后几乎持平

**(e)** 

GPT-2 XL若context_length 增大至 16384, 相比1024增大了16倍。投影/FFN/lm_head 随 T 乘 16、QKᵀ/attn@V 随 T² 乘 256。投影/FFN/lm_head 份额下降，attention scores 份额上升并成为最大头