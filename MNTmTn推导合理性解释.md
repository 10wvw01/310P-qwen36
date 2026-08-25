# M、N、Tm、Tn 推导合理性解释

本文专门解释前面容易混淆的几个量：

```text
M、N、K、Tm、Tn
```

目标模型是 Qwen3.6-35B-A3B-w8a8，硬件是 Ascend 310P3，TP=2。

## 1. 先说最重要的结论

当前结论需要分成两类看：

| 参数 | 当前结论 | 可信度 |
|---|---|---|
| Prefill 的 M=2048 | 一次确实处理完整 2048 个 token 时成立 | 高 |
| Decode 的 M=实际活跃 token 数 | 例如单请求通常是 1，多请求可能是 B | 高 |
| N=2048 | 由模型的 hidden size 决定 | 高 |
| Tm=128、Tn=256 | 当前只是一个代表性的估算值 | 尚未实测确认 |
| P=8、Split-K 后有效 P=4 | 是计算假设，最终要看实际 Kernel 调度 | 尚未完全确认 |

所以，前面的 M、N 推导是标准做法；前面表格中的 `128×256` 只能叫“理论评估点”，暂时不能叫“真实 Tile”。

## 2. `[x,y]` 到底表示什么

把一个矩阵写成：

```text
[x, y]
```

如果它表示 Transformer 的隐藏状态或输出结果，可以这样理解：

```text
x = 有多少个 token
y = 每个 token 有多少个特征
```

例如：

```text
H[2048, 2048]
```

表示：

- 有 2048 行，也就是 2048 个 token；
- 每一行有 2048 个特征，也就是每个 token 的 hidden size。

但 `[x,y]` 不是完整的 `o_proj/out_proj` 定义。完整的矩阵乘法应该写成：

```text
输入 X[M, K] × 权重 W[K, N] = 输出 Y[M, N]
```

其中：

- `M`：本次真正参与计算的 token 数；
- `K`：每个 token 在投影之前有多少个输入特征；
- `N`：每个 token 在投影之后有多少个输出特征。

因此，`[x,y]` 只是输出矩阵 `Y[M,N]` 的简写，其中 `x=M`、`y=N`。

补充一点：PyTorch 里权重有时会按 `[N,K]` 存储，计算时再转置；这只是存储方式不同，不改变“输入 K 维，输出 N 维”的逻辑。

## 3. 用一个很小的例子理解 TP=2

假设有：

```text
2 个 token
每个 token 有 4 个输入特征
每个 token 最终输出 2 个特征
TP=2
```

完整计算是：

```text
[2, 4] × [4, 2] = [2, 2]
```

TP=2 后，4 个输入特征被两张卡平分：

```text
卡 0：拿到 2 个输入特征
卡 1：拿到另外 2 个输入特征
```

于是：

```text
卡 0：[2, 2] × [2, 2] = [2, 2]   局部结果
卡 1：[2, 2] × [2, 2] = [2, 2]   局部结果
```

两张卡的结果不是拼接，而是相加：

```text
最终结果 = 卡 0 结果 + 卡 1 结果
         = [2, 2]
```

这就是 Row Parallel Linear 的标准做法：

```text
输入特征 K 被切开
输出特征 N 不切开
每张卡计算完整的输出列
最后 AllReduce 求和
```

这个规则只适用于当前讨论的 Row Parallel `o_proj/out_proj`，不是所有 GEMM 都这样切。若是 Column Parallel，通常会切输出维度 N，最后再拼接输出。

## 4. 映射到 Qwen3.6

### Full Attention 的 `o_proj`

模型配置为：

```text
16 个 Attention Head
每个 Head 256 维
```

所以投影前的总输入特征数是：

```text
16 × 256 = 4096
```

模型的 hidden size 是：

```text
2048
```

因此完整逻辑计算是：

```text
[M, 4096] × [4096, 2048] = [M, 2048]
```

TP=2 后，每张卡只处理 4096 的一半：

```text
4096 ÷ 2 = 2048
```

每张卡上的局部计算是：

```text
卡 0：[M, 2048] × [2048, 2048] → [M, 2048]
卡 1：[M, 2048] × [2048, 2048] → [M, 2048]
```

最后两张卡把两个 `[M,2048]` 结果相加，得到最终输出：

```text
[M, 2048]
```

### GDN/Linear Attention 的 `out_proj`

GDN 的投影前总输入特征数是：

```text
32 个 Value Head × 每个 128 维 = 4096
```

因此它和 Full Attention 的 `o_proj` 具有相同的逻辑形状：

```text
[M, 4096] × [4096, 2048] = [M, 2048]
```

## 5. 为什么 Prefill 的 M 是 2048

Prefill 一次处理完整 Prompt：

```text
batch = 1
Prompt 长度 = 2048 token
```

所以本次 GEMM 有 2048 行：

```text
M = 2048
```

Attention 输出投影的每卡 GEMM 是：

```text
[2048, 2048] × [2048, 2048] → [2048, 2048]
```

这里的 2048 行分别对应 2048 个 token。

注意：如果打开 chunked prefill，vLLM 可能把 2048 token 分成多个块。此时每次 GEMM 的 M 是当前块的 token 数，例如：

```text
本次只调度 512 token → M=512
```

## 6. 为什么 Decode 的 M 不是历史上下文长度

假设已经有 2048 个历史 token，Decode 当前只生成 1 个新 token：

```text
历史 KV Cache：2048 token
本次真正新计算：1 token
```

因此输出投影的 M 是：

```text
M=1
```

如果同时有 8 个请求，每个请求都生成 1 个新 token，则：

```text
M=8
```

所以 Decode 的 M 应理解为：

```text
当前这次 GEMM 实际同时处理的新 token 数
```

它不是历史上下文长度，也不是 KV Cache 长度。

## 7. Tm、Tn 和 M、N 不是一回事

`M、N` 是 GEMM 的逻辑输出矩阵大小；`Tm、Tn` 是硬件把这个输出矩阵切成小块时使用的大小。

例如：

```text
输出矩阵 C[2048,2048]
Tm=128，Tn=256
```

只是表示大矩阵被划成：

```text
M 方向：ceil(2048/128) = 16 块
N 方向：ceil(2048/256) = 8 块
总 Tile 数：16 × 8 = 128 块
```

原始矩阵仍然是：

```text
C[2048,2048]
```

不能把 `Tm=128` 当成 `M=128`，也不能把 `Tn=256` 当成 `N=256`。

## 8. 128×256 目前为什么只能算“估算值”

CANN 会根据下面这些因素，在运行时生成实际 Tiling：

- M、N、K 的大小；
- 数据类型；
- 输入/权重格式；
- 可使用的硬件核数；
- 是否 Split-K；
- 是否使用融合 Kernel。

因此，当前使用的：

```text
Tm=128，Tn=256
```

可以作为一个合理的预估点，但不能仅凭模型配置证明它就是实际值。

还要注意 CANN 里可能有两种不同层次的 Tile：

```text
singleCoreM/singleCoreN：一个 AI Core 负责的输出块
baseM/baseN：Core 内部一次矩阵乘指令使用的更小块
```

用于计算 Wave 的 `Tm/Tn`，通常应先确认是不是 `singleCoreM/singleCoreN`，不能直接把 `baseM/baseN` 当成 Wave Tile。

## 9. 如何证明真实的 Tm、Tn

### 第一步：用 profiler 验证 M、N

保持之前的数据采集参数，并打开：

```text
torch_profiler_record_shapes = true
```

分别采集：

1. 2048 token Prefill；
2. 单请求 Decode；
3. 多请求 Decode；
4. Split-K=2。

检查 `o_proj/out_proj` 的输入、权重、输出 Shape，以及 AllReduce 的输入 Shape。

预期结果：

```text
Prefill：输出 [2048,2048]
单请求 Decode：输出 [1,2048]
Decode batch=B：输出 [B,2048]
```

### 第二步：打印 CANN Tiling 参数

诊断运行时可以打开 CANN 日志：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

然后在日志中搜索：

```text
MatmulTiling
```

重点记录：

```text
M、N、Ka、Kb
singleCoreM、singleCoreN
baseM、baseN、baseK
usedCoreNum
Split-K 相关参数
```

如果日志显示：

```text
singleCoreM=128
singleCoreN=256
```

并且该值在同一场景的多个调用中稳定出现，才可以把当前 `Tm=128、Tn=256` 作为真实运行值。

### 第三步：重新计算 Wave

如果确认每个 Core 负责一个输出 Tile，则：

```text
Tm = singleCoreM
Tn = singleCoreN
```

然后计算：

```text
Tile 数 = ceil(M/Tm) × ceil(N/Tn)
Wave 数 = ceil(Tile 数 / usedCoreNum)
```

如果是 Split-K=2，还要确认两个 Core 是否真的共同完成一个输出 Tile，不能只凭“Split-K=2”直接把 8 个 Core 除以 2。

## 10. 当前结果应该怎样表述

在没有拿到真实 Tiling 日志前，建议这样写：

```text
M、N：根据模型结构、TP=2 和本次调度 token 数严格推导。
Tm、Tn：采用 128×256 作为代表性理论评估点，尚未实测确认。
P：采用 8 个 AIC 作为代表性并行度，最终以 usedCoreNum 为准。
Overlap 百分比：理论上限，不是实测端到端收益。
```

最完整的证明链是：

```text
模型配置
  → profiler 看到的矩阵 Shape
  → CANN 实际 Tiling 参数
  → 真实 Tile 数和 Wave 数
  → Kernel 时间线中的实际重叠
```

## 11. 记忆版

```text
M：这次有多少个 token 一起算
N：每个 token 最后输出多少个特征
K：投影前每个 token 有多少个输入特征
Tm/Tn：硬件把 C 矩阵切成多大一块
```

对当前 Attention 输出投影：

```text
M=2048（完整 2048-token Prefill）
N=2048（模型 hidden size）
K=4096（Attention 输出特征）
TP=2 后每卡 K=2048
```

而 `Tm=128、Tn=256` 要等实际 CANN Tiling 日志确认。
