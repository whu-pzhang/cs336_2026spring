
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