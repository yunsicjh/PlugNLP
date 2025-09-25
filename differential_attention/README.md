# MultiheadDiffAttn for NLP (Differential Transformer)

> 即插即用的差分注意力（Differential Attention）实现，集成 RoPE 位置编码、可选 Triton 加速、RMSNorm 与双路注意力差分机制，面向长上下文 & 噪声抑制型 NLP 任务。

## 1. 背景与来源

本模块参考 **Differential Transformer (ICLR 2025 Oral)** 的核心思想：使用两路（双分支）softmax 注意力图做差分，放大判别性结构，抑制冗余与噪声，从而得到更稀疏、更聚焦的注意力分布。

| 信息 | 内容                                                                    |
| ---- | ----------------------------------------------------------------------- |
| 论文 | Differential Transformer                                                |
| 作者 | Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, Furu Wei |
| 会议 | ICLR 2025 (Oral)                                                        |
| 链接 | [arXiv:2410.05258](https://doi.org/10.48550/arXiv.2410.05258)           |

## 2. 设计概述与实现细节

新版 `MultiheadDiffAttn` 代码相较最初简化版本的改动：

1. 集成 **RoPE**（Rotary Position Embedding），支持 interleaved 形式；无 Triton 时自动降级为 PyTorch 实现。
2. 采用 **两倍逻辑头数 (2 * num_heads)** 的 `Q/K` 结构：将每个逻辑 head 再拆为两个子 head（分支 a / b）以进行差分。
3. 使用 **差分 λ (lambda_full)**：由四组可学习参数 `(λ_q1, λ_k1, λ_q2, λ_k2)` 内积后指数化得到两个正标量，再相减 + 初始化偏置实现动态放缩。
4. 输出阶段用 `RMSNorm(2 * head_dim)` 做子头归一，有助于缓解差分后数值幅度不平衡。
5. 支持 **GQA / MHA 混合**：`num_kv_heads` 可与 `num_heads` 不同，通过 `repeat_kv` 复用 Key/Value。
6. 内部 `head_dim = embed_dim // num_heads // 2`（因 Q 拆成 2*num_heads 结构），须确保 `embed_dim % (num_heads * 2) == 0`。

### 2.1 计算流程（简化）

1. 线性投影：`Q ∈ R^{B,N,2H,d}, K ∈ R^{B,N,2H_k,d}, V ∈ R^{B,N,H_k,2d}`
2. 应用 RoPE：对 `Q,K` 的前 `2d` 维做旋转位置编码。
3. 复用扩展：`repeat_kv` 复制 KV 组以匹配 Query 头数。
4. 注意力计算：`A = softmax(Q_a K_a^T / sqrt(d))` 与 `B = softmax(Q_b K_b^T / sqrt(d))`（在实现里通过 reshape / view 一次性得到，再切分）。
5. 差分融合：`A' = A - λ * B`，其中 `λ = exp(<λ_q1,λ_k1>) - exp(<λ_q2,λ_k2>) + λ_init`。
6. 加权求和：`O = A' V`，再经 RMSNorm + 线性映射返回原维度。

> 该结构不是概率注意力（差分后可出现负值），更接近“对比权重混合”；在需要保持严格概率语义的场景（如可视化权重解释）可选再做正则化。

## 3. NLP 任务中的优势

| 维度                | 优势说明                                                                      |
| ------------------- | ----------------------------------------------------------------------------- |
| 噪声抑制            | 差分抵消第二分支常见/背景注意力，突出一阶判别模式（实体、触发词、关键段落）。 |
| 稀疏化              | 差分后很多无信息区域趋近 0（甚至负值），对下游 MLP 更友好，减少干扰。         |
| 长上下文            | 结合 RoPE，全局相对位置信息保留，差分提升跨段筛选能力，适合文档 QA / 摘要。   |
| 幻觉抑制            | 对生成式任务，抑制“平均扩散注意力”导致的虚构事实，提升忠实度。                |
| In-Context Learning | 对提示中关键示例位置更敏感，顺序扰动鲁棒性提升。                              |
| GQA 支持            | 降低 KV 投影开销，使长序列推理更经济。                                        |
| 可扩展性            | 与普通注意力层交替堆叠可形成“精细对齐 + 差分突显”的互补结构。                 |

## 4. 与标准多头注意力对比

| 方面     | 标准 MHA          | Differential MultiheadDiffAttn |
| -------- | ----------------- | ------------------------------ |
| 头结构   | H                 | 逻辑 H，对应 2H 子头差分       |
| 权重     | 单一 softmax 概率 | 差分 (A - λ B)，非概率，可为负 |
| 关注模式 | 全局扩散或尖峰    | 更倾向稀疏 + 高对比            |
| 可解释性 | 概率易解释        | 需看 A / B 分布与 λ            |
| RoPE     | 可选              | 内置支持（fallback 可用）      |
| GQA      | 额外设计          | 直接支持 `num_kv_heads`        |
| 输出稳定 | 依赖 LayerNorm    | RMSNorm + 差分缩放             |

## 5. 使用方法

### 5.1 安装依赖

基础依赖：
```bash
pip install torch
```
可选（Triton 加速 RoPE；仅 GPU 且特定环境支持）：
```bash
pip install triton
```

### 5.2 生成 RoPE 参数

`generate_rope_embeddings(seq_len, head_dim)` 的第二个参数必须是 **内部实际的 per-branch head_dim**，即：

```
internal_head_dim = embed_dim // num_heads // 2
```

不要传入 `internal_head_dim * 2`，否则维度会错配（因为函数内部会按偶数索引再除 2 生成 cos/sin）。

```python
from differential_attention.diffattn_nlp import generate_rope_embeddings

seq_len = 1024
head_dim = 64  # 必须与 internal_head_dim 一致
cos, sin = generate_rope_embeddings(seq_len, head_dim, device='cuda')
```

### 5.3 初始化与前向

```python
from differential_attention.diffattn_nlp import MultiheadDiffAttn, generate_rope_embeddings
import torch

embed_dim = 1024          # 必须满足 embed_dim % (num_heads * 2) == 0
num_heads = 8             # 逻辑头 H，内部 Q/K 拆成 2H 子头差分
depth = 0                 # 当前层索引（影响 λ 初始化）
model = MultiheadDiffAttn(embed_dim, depth, num_heads).cuda()

# 验证内部 head_dim 公式
assert model.embed_dim % (model.num_heads * 2) == 0
print("internal head_dim =", model.head_dim)  # 这里应为 64

seq_len = 512
# 传入的第二个参数就是 internal head_dim（不要 *2）
cos, sin = generate_rope_embeddings(seq_len, model.head_dim, device='cuda')
x = torch.randn(2, seq_len, embed_dim).cuda()
out = model(x, (cos, sin))            # out: [2, seq_len, embed_dim]
```

### 5.4 与注意力 Mask

当前实现：
- 若 `attn_mask is None`，内部默认构造 **自回归 (causal) 上三角 mask**。
- 若需要 padding mask，可自行构造 additive mask（被屏蔽位置 = `-inf`）。

支持的 `attn_mask` 形状（会自动广播到 `(B, H, tgt_len, src_len)`）：
- `(tgt_len, src_len)`
- `(1, 1, tgt_len, src_len)`
- `(B, 1, 1, src_len)` （典型 padding 掩码）

例如：

```python
pad_mask = (input_ids != pad_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
add_mask = (~pad_mask).float() * float('-inf')              # 被屏蔽位置 -inf
out = model(x, (cos, sin), attn_mask=add_mask)
```

### 5.5 与标准 Transformer 层混合

建议模式：
```
[ (StdAttention) → (DiffAttn) → (StdAttention) … ]
```
或每 3~4 层插入 1 层差分层，用于突出全局判别特征。

## 6. 参数解释

| 参数             | 含义           | 关键约束 / 备注                         |
| ---------------- | -------------- | --------------------------------------- |
| `embed_dim`      | 模型隐藏维度   | 需满足 `embed_dim % (num_heads*2) == 0` |
| `num_heads`      | 逻辑多头数     | 内部 Q/K 拆成 `2*num_heads` 子头        |
| `num_kv_heads`   | (可选) KV 头数 | 默认为 `num_heads`；GQA 时 < num_heads  |
| `depth`          | 层索引         | 控制 λ 初始偏置（浅层更保守）           |
| `lambda_*`       | 四组向量参数   | 指数内积后形成两个缩放分量 λ1/λ2        |
| `lambda_init_fn` | λ 基值函数     | 随深度递减差异，避免初期过拟合          |
| `n_rep`          | KV 复用倍数    | `num_heads // num_kv_heads`             |
| `head_dim`       | 子头维度       | `embed_dim // num_heads // 2`           |
| `RMSNorm`        | 子头归一       | 放在差分后的加权输出上                  |
| RoPE             | 位置编码       | interleaved 形式；Triton 可加速         |

## 7. 调参与实践建议

| 目标       | 建议策略                                                  |
| ---------- | --------------------------------------------------------- |
| 训练不稳定 | 降低学习率；对 `lambda_*` 参数设置更小初始 std（如 0.05） |
| 差分振荡   | clamp λ：`lambda_full = torch.clamp(lambda_full, -5, 5)`  |
| 解释需求   | 单独观察两路注意力：在 `view` 切分后保存 `attn_a, attn_b` |
| 长序列     | 与线性/稀疏注意力交替，降低 O(N²) 成本集中在关键层        |
| 幻觉抑制   | 在生成模型 decoder 中替换中后段若干层                     |
| In-context | 在 prompt 位置添加少量显式 anchor token，加强差分对比     |

## 8. 限制与注意事项

1. 差分后的权重非概率分布；若下游需要概率（如可视化）需再正则化。
2. 可能出现负注意力 → 值向量的“反相整合”，理论上可增强对比，但也会带来梯度震荡；必要时加 `attn = attn / (attn.abs().sum(-1, keepdim=True)+1e-9)`。
3. 当前实现中 `attn_mask` 默认自回归；如果用于 **encoder-only** 模型记得显式传递 `attn_mask=None` 或构造 padding mask。
4. λ 采用指数内积，极端情况下可能溢出；可选改为 `F.softplus` 或加入 clamp。
5. Triton 非必需；安装失败/CPU 环境下自动使用 PyTorch fallback；频率极高调用下 GPU Triton 会更省时。
6. `head_dim` 不宜过小（< 16）否则差分表达力下降；过大则显存开销升高。

## 9. 扩展方向（Roadmap 提示）

- 支持混合差分策略：`A - λ1*B - λ2*C`（多分支抑制）。
- 引入结构化稀疏：在差分后对权重 top-k 截断。
- 加入可学习门控决定是否启用差分（Dynamic DiffAttn）。
- 与频域/小波混合（可与本仓库 SPECTRE 组件交替）。

## 10. 参考文献

1. Differential Transformer. ICLR 2025 (Oral). arXiv:2410.05258.
2. RoFormer: Enhanced Transformer with Rotary Position Embedding.
3. GQA (Grouped Query Attention) 相关实现理念（提高 KV 复用效率）。

## 11. 版权与声明

本实现为参考论文思想的第三方工程化适配，便于实验验证与模块化复用；与原论文官方实现可能存在实现细节差异。使用时请同时引用原论文。

---
最后更新：2025-09-25