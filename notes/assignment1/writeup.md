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
