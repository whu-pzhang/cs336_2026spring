
# Assignment 1 学习笔记

## BPE Tokenizer 优化：最小堆与双向链表

### 1. BPE 编码到底在做什么

对一个 GPT-2 正则表达式产生的 pre-token，先把 UTF-8 字节转换成初始 token ID，然后重复执行：

1. 查看所有相邻 token pair；
2. 找出 merge rank 最小的 pair；
3. 把这个 pair 合并成一个新 token；
4. 直到没有可合并的 pair。

例如，假设当前序列是：

```text
[a, b, c, d]
```

如果 `(b, c)` 的 rank 最小，那么合并过程是：

```text
[a, b, c, d] -> [a, bc, d]
```

这里的 rank 是训练时 merge 的顺序，不是当前文本中 pair 的出现频率。rank 越小，表示这个 pair 越应该优先合并。

### 2. 朴素实现的两个瓶颈

最直接的代码可以用一个 Python `list` 保存当前 token：

```python
while True:
    扫描 parts 中的所有相邻 pair
    找出 rank 最小的 pair
    删除右侧 token，并把合并结果写回左侧
```

这种实现容易理解，但每次合并都有两个成本：

- 要扫描整个 `parts`，才能找到 rank 最小的 pair；
- 删除 list 中间的元素时，后面的元素需要整体移动。

如果一个 pre-token 有 `n` 个字节，最多需要进行 `n - 1` 次合并，那么整体成本大致是 `O(n^2)`。短文本上没有问题，但在大量数据上，重复扫描和移动会成为瓶颈。

### 3. 为什么使用双向链表

优化版仍然需要表示“相邻 token”，但不再直接删除 list 元素，而是用三个数组模拟双向链表：

```python
prev[i]   # 节点 i 左边的节点下标
next_[i]  # 节点 i 右边的节点下标
alive[i]  # 节点 i 是否仍然有效
```

例如初始序列 `[a, b, c, d]` 可以表示为：

```text
a <-> b <-> c <-> d
```

合并 `b` 和 `c` 时，不需要移动 `d`，只需做局部更新：

```text
a <-> b <-> c <-> d
        \合并/
a <-> bc <-> d
```

代码层面主要是：

```python
token_ids[b] = merged_id
alive[c] = False
next_[b] = next_[c]
prev[next_[c]] = b
```

因此，一次合并只会修改被合并节点及其邻居，避免了 Python list 的整体搬移。

这里的“链表”不是节点对象组成的传统链表，而是用数组下标作为节点地址。这样既保留了链表的局部更新特性，又避免了为每个节点创建额外 Python 对象。

### 4. 为什么使用最小堆

每次 BPE 合并都需要找到 rank 最小的相邻 pair。优化版把当前可能合并的 pair 放进最小堆，堆元素包含：

```text
(rank, left_index, left_id, right_id)
```

由于堆按第一个字段排序，`heappop` 可以直接给出 rank 最小的候选，不需要再次扫描全部相邻 pair。

合并一个 pair 后，真正可能发生变化的只有两个位置：

```text
旧左邻居 <-> 新合并节点 <-> 旧右邻居
```

所以只需把这两个新相邻 pair 加回堆中，而不是重新建立整个候选集合。

### 5. 为什么堆里允许失效候选

一个 pair 被放入堆后，邻居可能在它之前被合并。例如堆中原来有：

```text
(a, b), (b, c), (c, d)
```

如果先合并 `(b, c)`，那么 `(a, b)` 和 `(c, d)` 中至少有一个节点已经失效。主动从堆中删除所有相关旧候选需要额外维护索引，代码反而更复杂。

因此实现采用“惰性删除”：候选弹出时再检查它是否仍然有效。需要同时确认：

1. 左右节点都还处于 `alive` 状态；
2. 当前节点 ID 仍等于候选记录中的 `left_id` / `right_id`；
3. 这个 pair 当前的 rank 仍等于候选记录中的 rank。

任一条件不满足，就丢弃这个候选，继续弹出下一个。这样可以用少量校验换取更简单的堆维护逻辑。

### 6. 优化后的复杂度

设 pre-token 初始有 `n` 个字节：

- 初始化相邻 pair 并建堆：约 `O(n log n)`；
- 每次合并只更新常数个链表邻居，并执行有限次堆操作：约 `O(log n)`；
- 总共最多 `n - 1` 次合并，因此整体约为 `O(n log n)`。

朴素 list 版本约为 `O(n^2)`，所以最小堆 + 链表在长 pre-token 或大量重复编码时更有优势。代价是需要额外数组和堆空间，代码中的 `alive` 与失效候选检查也增加了理解成本。

### 7. Pretoken 缓存的动机与取舍

语料中常会反复出现相同的 pre-token，例如 `the`、`and` 或常见标点。对同一个 pre-token 重复执行 BPE 没有必要，因此 `Tokenizer` 会缓存：

```text
原始 pre-token bytes -> 最终 token ID tuple
```

缓存命中时直接返回结果；达到容量上限后，移除最久没有使用的条目。缓存能显著减少重复计算，但会占用内存，所以设置了固定容量 `100_000`。如果处理的是几乎没有重复词的语料，缓存收益会降低，内存成本仍然存在。

### 8. 代码阅读顺序

建议按下面的顺序阅读 `assignment1-basics/cs336_basics/tokenizer.py`：

1. `GPT2_SPLIT_PATTERN`：理解 pre-token 的边界；
2. `pre_tokenize`：理解训练数据如何变成 byte tuple 计数；
3. `train_bpe`：理解 merge pair 如何从高频统计中产生；
4. `Tokenizer.__init__`：理解词表、merge rank 和 special token 的预计算；
5. `_bpe_encode_pretoken`：先看朴素 BPE 逻辑，再分别理解 heap、链表和缓存；
6. `encode` / `decode`：理解普通文本与 special token 如何组合。

## Cross Entropy 推导

交叉熵定义如下：

$$
\mathcal{L} = - \log \left( \frac{e^{z_y}}{\sum_{j=1}^{V} e^{z_j}} \right)
$$

其中 $z\in \mathrm{R}^V$ 是 logits 向量，$y$ 是目标索引，$V$是词表大小。

直接计算会面临两个数值问题：

1. **上溢(overflow)**: 如果某个 $z_j$ 很大，则 $e^{z_j}$ 可能超出浮点数范围（`inf`）
2. **下溢(underflow)**: 如果 $z_y$ 远小于其他 logits，则 softmax 概率可能接近零，取对数后得到 `-inf`

利用对数性质，对上式化简改写为：

$$
\mathcal{L} = -z_y + \log \left( \sum_{i=1}^{V} e^{z_j} \right)
$$

到这里后，关键就是数值稳定地计算 $\log \left( \sum_{i=1}^{V} e^{z_j} \right)$，这个函数叫 **log-sum-exp**

令 $m = \max_{j} z_j$，则：

$$
\log \left( \sum_{i=1}^{V} e^{z_j} \right) = \log \left( e^m \sum_{j=1}^V e^{z_j - m} \right) = m + \log \left(\sum_{j=1}^V e^{z_j - m} \right)
$$

因为 $z_j - m \le 0$，所以所有指数项都在 $[0,1]$ 范围内，彻底避免了overflow。对求和取对数，因为至少有一项为1，当 $z_j = m$ 时，其值也远大于 `-inf`，也避免了underflow。

将上式代入，最终得到交叉熵损失如下：

$$
\mathcal{L} = -z_y + m + \log \left(\sum_{j=1}^V e^{z_j - m} \right)
$$

或者等价的，先计算 log-softmax（实际代码实现方式）:

$$
\log(\text{softmax}(z)_i)  = z_j - m - \log \left(\sum_{k=1}^V e^{z_k - m} \right)
$$

然后取负。
