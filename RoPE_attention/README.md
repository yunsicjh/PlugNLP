# StandardAttentionWithRoPE （RoPE 增强多头注意力）

> 基于 RoFormer 论文 (RoPE: Rotary Position Embedding) 的即插即用注意力模块，实现标准多头自注意力 + RoPE 相对位置编码，支持返回注意力权重、跨注意力场景与多种 mask 形式。

## 1. 背景与论文信息

| 项目  | 内容                                                                  |
| ----- | --------------------------------------------------------------------- |
| 论文  | RoFormer: Enhanced Transformer with Rotary Position Embedding         |
| 作者  | Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu |
| ArXiv | [arXiv:2104.09864](https://doi.org/10.48550/arXiv.2104.09864)         |
| 领域  | 相对位置编码 / 长序列建模                                             |

**核心思想**：用频率依赖的二维旋转矩阵对每个 head 的偶/奇位通道成对旋转，将绝对位置编码为向量相位，同时在点积中隐式引入相对位置信息，使模型：

- 拥有随距离衰减的注意力偏置
- 在扩展序列长度时更平滑（外推能力好）
- 易于与多种注意力变体（线性、稀疏、增量）组合

## 2. 模块特性概览

| 特性      | 描述                                                             |
| --------- | ---------------------------------------------------------------- |
| RoPE 支持 | 原生偶/奇通道 pairwise 旋转实现（非简单前后切半）                |
| 可扩展    | 同时支持自注意力与交叉注意力（query/key/value 可不同）           |
| 形状友好  | 输入输出形状均为 (B, N, D)，易集成到现有 Transformer Block       |
| 返回权重  | `return_attn=True` 可取 softmax 后注意力权重（调试/可视化）      |
| 掉落支持  | 权重级 dropout, 与标准 MHA 保持一致                              |
| 掩码兼容  | 支持 additive mask (填 -inf)，以及 broadcast 到 (B,H,N,N) 的张量 |

## 3. 使用优势（对 NLP 场景）

| 场景                           | 优势说明                                                                  |
| ------------------------------ | ------------------------------------------------------------------------- |
| 长文本分类 / QA / 摘要         | 相对距离信息显式融入注意力相似度，远距离 token 依赖自然衰减，减少噪声干扰 |
| 生成式模型 (LLM)               | 增强 extrapolation（训练 2K 推理 4K/8K 更稳定）                           |
| 检索 / 排序                    | 位置偏置不需额外参数表，可减少参数并保持对齐一致性                        |
| 结构化序列（代码、日志、表格） | 频率分量旋转对重复模式的相似性度量更鲁棒                                  |
| 轻量部署                       | 无需额外可学习参数（除线性层），RoPE 仅是前向计算                         |

## 4. 快速开始

### 4.1 安装依赖

```bash
pip install torch
```

### 4.2 生成 RoPE 参数

```python
from RoPE_attention.rope_attn import generate_rope_embeddings

seq_len = 1024
head_dim = 64  # = embed_dim // num_heads，且必须为偶数
cos, sin = generate_rope_embeddings(seq_len, head_dim, device='cuda')
```

### 4.3 初始化与前向调用

```python
from RoPE_attention.rope_attn import StandardAttentionWithRoPE, generate_rope_embeddings
import torch

embed_dim = 512
num_heads = 8
attn = StandardAttentionWithRoPE(embed_dim, num_heads, dropout=0.1).cuda()

B, N = 2, 256
x = torch.randn(B, N, embed_dim, device='cuda')
cos, sin = generate_rope_embeddings(N, embed_dim // num_heads, device='cuda')

out, weights = attn(x, x, x, rope_embeddings=(cos, sin), return_attn=True)
print(out.shape, weights.shape)  # (B,N,D), (B,H,N,N)
```

### 4.4 Causal / Padding 掩码示例

```python
import torch

# 构造 causal mask: 上三角为 -inf
causal = torch.full((N, N), float('-inf'))
causal = torch.triu(causal, diagonal=1)  # (N,N)
causal = causal.unsqueeze(0).unsqueeze(0)  # (1,1,N,N) 可广播

# Padding mask: 1=keep, 0=pad
pad_mask = (torch.randint(0, 10, (B, N)) > 1).to(torch.bool)  # 伪造样例
add_mask = (~pad_mask).unsqueeze(1).unsqueeze(2) * float('-inf')  # (B,1,1,N)

out = attn(x, x, x, rope_embeddings=(cos, sin), attn_mask=causal)
out_pad = attn(x, x, x, rope_embeddings=(cos, sin), attn_mask=add_mask)
```

## 5. API 说明

### 5.1 类: `StandardAttentionWithRoPE`

| 参数        | 说明                                        |
| ----------- | ------------------------------------------- |
| `embed_dim` | 总隐藏维度 D                                |
| `num_heads` | 头数 H，需满足 `embed_dim % num_heads == 0` |
| `dropout`   | 注意力权重 dropout 概率                     |

### 5.2 `forward` 参数

| 名称              | 形状                     | 说明                                                         |
| ----------------- | ------------------------ | ------------------------------------------------------------ |
| `query`           | (B,Nq,D)                 | 查询序列                                                     |
| `key`             | (B,Nk,D)                 | 键序列                                                       |
| `value`           | (B,Nk,D)                 | 值序列                                                       |
| `rope_embeddings` | (cos,sin)                | 由 `generate_rope_embeddings` 生成；cos/sin 形状 (>=N, Dh/2) |
| `attn_mask`       | broadcast 到 (B,H,Nq,Nk) | additive mask；被屏蔽位置加 -inf                             |
| `return_attn`     | bool                     | 是否返回注意力权重                                           |

### 5.3 返回值

`out` 或 `(out, attn_weights)`：
| 名称           | 形状        | 含义                                          |
| -------------- | ----------- | --------------------------------------------- |
| `out`          | (B,Nq,D)    | 输出序列表示                                  |
| `attn_weights` | (B,H,Nq,Nk) | softmax 后注意力权重（`return_attn=True` 时） |

## 6. 实现细节 & 正确性说明

1. RoPE 采用 **偶/奇通道成对旋转**：将 head_dim 重塑为 (Dh/2, 2)，分别视为 (x_even,x_odd)。
2. 当提供的 cos/sin 长度 < 当前序列长度时，会使用最后一行重复填充（外推场景保持平稳）。
3. 支持不同 `query` 与 `key/value` 长度（交叉注意力）；需保证 RoPE 至少覆盖 `max(Nq,Nk)`。
4. 不引入额外可训练参数；RoPE 完全是前向可微算子（对 cos/sin 无梯度）。

## 7. 调参与实践建议

| 目标                      | 建议                                                           |
| ------------------------- | -------------------------------------------------------------- |
| 更长上下文外推            | 训练阶段提前用略大 seq_len 生成 cos/sin（例如训练 2K 生成 4K） |
| 低显存                    | 降低 num_heads，保持 head_dim ≥ 32 以保证旋转分辨率            |
| 可视化                    | 传 `return_attn=True`，并对 `attn_weights.mean(1)` 做热力图    |
| 与差分注意力/频域模块混合 | 在堆叠中保留首末层标准 RoPE 注意力，提升对齐稳定性             |

## 8. 常见问题 (FAQ)

1. Q: head_dim 为什么必须是偶数？  
	A: RoPE 需要将通道对 (even, odd) 组成二维旋转。
2. Q: 如何与 FlashAttention 结合？  
	A: 在调用高性能 kernel 前先对 Q/K 做 RoPE 变换，再喂入 kernel。
3. Q: 生成阶段如何增量缓存？  
	A: 仅需对新增 token 的 Q/K 应用对应位置索引的 cos/sin 并追加 KV 缓存。
4. Q: cos/sin 可以学习吗？  
	A: 原始 RoPE 不建议；若需可学习相对偏置，可在 scores 上加额外可学习 bias。

## 9. 兼容与限制

| 项目         | 说明                                                       |
| ------------ | ---------------------------------------------------------- |
| 支持设备     | CPU / GPU 均可（无专用 CUDA kernel，易移植）               |
| 序列外推     | 可适度外推（>2× 需谨慎，建议结合插值或分段策略）           |
| 混合精度     | 支持 FP16/BF16；内部未使用不稳定的 exp/log 序列操作        |
| 旋转维度裁剪 | 当前对整个 head_dim 旋转；如需部分旋转可改写 `_apply_rope` |

## 10. 参考文献

1. Jianlin Su et al. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.
2. GPT-NeoX / LLaMA 等开源实现中的 RoPE 变体实践。

## 11. 文档作者与开放版权

**作者**: yunsicjh  
**本模块实现**: 基于公开论文思想的独立重写与工程化封装。  
**协议**: 继承仓库全局 LICENSE (Apache-2.0)。  

使用/引用时请注明：本仓库地址 + RoFormer 原论文。欢迎 Issue / PR 讨论改进（如局部旋转、缩放因子、外推增强等）。

---
最后更新：2025-09-25
