# SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized Batch Normalization

论文地址：[https://arxiv.org/abs/2405.11582](https://arxiv.org/abs/2405.11582)  
会议收录：ICML 2024  
代码仓库：[https://github.com/xinghaochen/SLAB](https://github.com/xinghaochen/SLAB)

---

## 模块简介

`RepBN_NLP` 是基于 SLAB 论文提出的 Progressive Re-parameterized Batch Normalization（PRepBN）思想，专为自然语言处理（NLP）任务设计的批归一化层。该模块结合了传统 BatchNorm1d 和可学习残差缩放参数 `alpha`，实现归一化与原始特征的自适应融合，适用于 Transformer、RNN、BERT 等模型的 Token 表示或隐藏状态归一化。

---

## 主要优点

- **自适应归一化强度**  
  通过可学习参数 `alpha`，模型能够自动调整归一化输出和原始特征之间的比例。
- **提升训练稳定性与泛化能力**  
  归一化助力模型收敛与泛化，残差路径避免信息损失。
- **易于集成**  
  接口与标准 PyTorch 模块一致，可直接替换 LayerNorm、BatchNorm1d 等现有归一化层。
- **灵活性高**  
  当 `alpha` 学习为 0 时可退化为普通归一化，为 1 时可忽略归一化，仅保留原始特征。

---

## 适用场景

- Transformer/BERT/自注意力网络的输入特征归一化
- RNN/LSTM/GRU 等序列模型的输出归一化
- 文本分类、序列标注、问答、文本生成等各类NLP任务

---

## 使用方法

### 1. 安装依赖

确保你的环境中已安装 PyTorch 1.6 及以上版本。

```bash
pip install torch
```

### 2. 模块引入与初始化

```python
from repbn_nlp import RepBN_NLP

# 假设Embedding或隐藏层维度为 embed_dim
repbn = RepBN_NLP(embed_dim=768)
```

### 3. 前向传播调用

输入张量 shape 应为 `[batch_size, seq_len, embed_dim]`：

```python
x = repbn(x)
```

### 4. 替换LayerNorm/BatchNorm用法示例

例如在自定义 Transformer Encoder 层中：

```python
import torch.nn as nn
from repbn_nlp import RepBN_NLP

class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, ...):
        super().__init__()
        self.norm1 = RepBN_NLP(embed_dim)
        # 原本可为 LayerNorm(embed_dim)

    def forward(self, x, ...):
        x = self.norm1(x)
        ...
```

---

## 模块实现原理

核心思想是：  
- 先将输入 `[batch, seq_len, embed_dim]` 转为 `[batch, embed_dim, seq_len]`，对 `embed_dim` 维做归一化；
- 输出为 `BatchNorm(x) + alpha * x`，融合归一化和原始特征；
- 再转回原始 shape。

### 代码实现

```python
import torch
import torch.nn as nn

class RepBN_NLP(nn.Module):
    """
    RepBN_NLP applies BatchNorm1d along the embedding/features dimension for NLP tasks.
    It also introduces a learnable alpha parameter for residual scaling.
    Input: Tensor of shape [batch_size, seq_len, embed_dim]
    """
    def __init__(self, embed_dim):
        super(RepBN_NLP, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        x = x.transpose(1, 2)             # [batch_size, embed_dim, seq_len]
        x_bn = self.bn(x)                 # [batch_size, embed_dim, seq_len]
        x = x_bn + self.alpha * x         # Residual scaling
        x = x.transpose(1, 2)             # [batch_size, seq_len, embed_dim]
        return x
```

---

## 常见问题

### Q1: 与 LayerNorm 有何不同？

LayerNorm 归一化轴为每个 token 的所有特征，BatchNorm1d 归一化轴为所有 token 的同一特征。`RepBN_NLP` 兼具批归一化与残差能力，适合大批量、长序列的 NLP 场景。

### Q2: 推理模式下表现？

推理时，BatchNorm1d 使用训练期统计均值和方差，`alpha` 仍可调节残差比例。

---

## 参考资料

- 论文：[SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized Batch Normalization](https://arxiv.org/abs/2405.11582) （ICML 2024）
- 代码仓库：[https://github.com/xinghaochen/SLAB](https://github.com/xinghaochen/SLAB)

---

## 维护者

- 作者: yunsi
- 本文档最后更新：2025-09-24

如有问题欢迎提issue或PR交流。