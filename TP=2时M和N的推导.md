# TP=2 时 M 和 N 的推导

## 结论先说

这里讨论的是 Attention 输出投影：

```text
out_proj/o_proj GEMM → TP AllReduce
```

假设：

```text
batch = 1
输入序列长度 = 2048 token
TP = 2
```

对于 Prefill：

```text
A[2048,2048] × B[2048,2048] → C[2048,2048]
```

其中：

```text
M = 2048
N = 2048
```

这里的 `M=2048` 表示一次有 2048 个 token 需要计算，并不是 Tile 大小、Attention Head 数或硬件内部块大小。

## 1. GEMM 中 M、K、N 分别代表什么

矩阵乘法统一写成：

```text
A[M, K] × B[K, N] = C[M, N]
```

含义是：

- `M`：有多少行输入样本
- `K`：每个输入样本有多少个特征
- `N`：每个输出样本有多少个特征

在 Transformer 中，通常：

```text
一行 = 一个 token 的隐藏向量
```

因此，输入有 2048 个 token，就有 2048 行：

```text
M = 2048
```

## 2. 2048 个 token 如何变成矩阵

输入 token：

```text
[t1, t2, t3, ..., t2048]
```

经过 Embedding 和前面的 Attention/Linear Attention 后，得到隐藏状态：

```text
H[2048, 2048]
```

这里两个 `2048` 的含义不同：

```text
H[2048, 2048]
  ↑       ↑
  M       hidden_size
```

- 第一个 `2048`：有 2048 个 token，也就是 `M=2048`
- 第二个 `2048`：每个 token 的隐藏维度，即 `hidden_size=2048`

注意：这只是本模型中“序列长度”和“hidden_size”恰好都为 2048，并不是同一个概念。

例如，如果模型的 hidden size 是 4096，但输入仍然是 2048 token，那么矩阵仍然是：

```text
H[2048, 4096]
```

此时 `M` 仍然是 2048。

## 3. Full Attention 的 o_proj

Full Attention 有：

```text
num_attention_heads = 16
head_dim = 256
```

每个 token 在进入 `o_proj` 前的总维度是：

```text
16 × 256 = 4096
```

所以单卡 TP=2 时，每张卡只处理一半：

```text
4096 / 2 = 2048
```

因此，每张卡上的逻辑 GEMM 是：

```text
A[2048, 2048] × B[2048, 2048]
```

结果：

```text
C[2048, 2048]
```

具体含义：

```text
A 的 2048 行 = 2048 个 token
A 的 2048 列 = 当前 TP rank 上的局部 Attention 特征
C 的 2048 行 = 2048 个 token
C 的 2048 列 = hidden_size=2048
```

两个 TP rank 分别算出一个局部结果，随后进行 AllReduce：

```text
C_rank0[2048,2048]
C_rank1[2048,2048]
             ↓ AllReduce(sum)
C[2048,2048]
```

所以这里的 M 仍然是 2048。

## 4. Linear/GDN Attention 的 out_proj

GDN 的 Value 维度是：

```text
linear_num_value_heads = 32
linear_value_head_dim = 128
```

总维度：

```text
32 × 128 = 4096
```

TP=2 后，每张卡处理：

```text
4096 / 2 = 2048
```

因此 GDN 的输出投影也是：

```text
A[2048,2048] × B[2048,2048]
→ C[2048,2048]
```

所以 Full Attention 和 GDN Attention 的候选融合算子对，M、N 完全相同。

## 5. 为什么不是 44 或 64

### 不是 44

如果输入确实是一个 batch、2048 token 的 Prefill，那么：

```text
M = 2048
```

只有在以下情况，M 才可能是 44：

- 实际输入只有 44 个 token
- vLLM 当前 chunk 只调度了 44 个 token
- 44 是某个中间 kernel 的局部 batch/token 数

44 不是 Qwen3.6 这个输出投影 GEMM 的固定模型维度。

### 不是 64

64 可能来自以下概念，但都不是当前 GEMM 的逻辑 M：

- 硬件内部 Tile 大小，例如 `Tm=64`
- 某个算子的 `baseM=64`
- 某个 batch 或 scheduler chunk 恰好有 64 个 token
- 某个内部向量/矩阵分块大小

例如，如果硬件使用：

```text
Tm = 64
```

那么只是把 M 方向切成：

```text
ceil(2048 / 64) = 32 个 Tile
```

但原始 GEMM 仍然是：

```text
C[2048,2048]
```

不能把 Tile 大小 64 当成矩阵的 M。

## 6. Prefill 和 Decode 的区别

### Prefill

一次性处理完整的 2048-token Prompt：

```text
batch = 1
seq_len = 2048
M = batch × seq_len = 1 × 2048 = 2048
```

因此：

```text
C[2048,2048]
```

### Decode

Decode 阶段虽然已经有 2048 个历史 token，但每一步通常只计算一个新 token：

```text
batch = 1
新生成 token 数 = 1
M = 1
```

因此 Decode 输出投影是：

```text
A[1,2048] × B[2048,2048]
→ C[1,2048]
```

历史 2048-token KV Cache 会影响 Attention 读取和计算，但不会让输出投影的 M 变成 2048。

如果 Decode 同时有 64 个请求，那么：

```text
M = 64
```

因此：

```text
M = 当前实际调度的 token 数
```

而不是固定等于上下文长度。

## 7. 最终记忆方式

对于当前讨论的 Attention 输出投影：

```text
M = 当前要同时处理的 token 数
N = hidden_size = 2048
K = Attention 输出特征维度，TP 后为 2048
```

单请求、2048-token Prefill：

```text
A[2048,2048] × B[2048,2048]
→ C[2048,2048]
```

单请求 Decode：

```text
A[1,2048] × B[2048,2048]
→ C[1,2048]
```

因此，**2048 是 Prefill 的逻辑矩阵 M；64 或 44 只有在实际 batch、实际调度 chunk 或硬件 Tile 恰好为这些数时才会出现，不能直接替代 M。**
