# SpectreBlockNLP - 频域混合自注意力替换模块

---

## 论文出处

- **论文题目**：The FFT Strikes Back: An Efficient Alternative to Self-Attention
- **发表时间**：2024年2月
- **发表平台**：arXiv预印本（论文编号：2502.18394）
- **论文链接**：[arXiv 公开页](https://arxiv.org/html/2502.18394v1)
- **PDF下载**：[ResearchGate PDF](https://www.researchgate.net/profile/Jacob-Fein-Ashley/publication/389351112_The_FFT_Strikes_Back_An_Efficient_Alternative_to_Self-Attention/links/6812b090bfbe974b23bf6858/The-FFT-Strikes-Back-An-Efficient-Alternative-to-Self-Attention.pdf)
- **Huggingface论文索引**：[https://huggingface.co/papers/2502.18394](https://huggingface.co/papers/2502.18394)

---

## 模块简介

SpectreBlockNLP 是一个专为 NLP 任务设计的频域混合 Token Mixer，可无缝替换 Transformer 结构中的自注意力层（self-attention）。该模块利用快速傅里叶变换（FFT）在频域进行 token 混合，实现全局信息交互，并通过门控机制和可选小波（wavelet）增强，显著提升长序列处理能力和推理效率。

---

## 优点分析

1. **高效性**  
   - 采用 FFT，序列混合复杂度从 O(N²) 降至 O(N log N)，大幅降低算力消耗，适合长文本、文档级NLP任务。

2. **全局感知能力强**  
   - 频域处理可天然捕捉全局依赖，提升长距离信息建模效果，无需传统自注意力的显式对齐计算。

3. **可插拔设计**  
   - 接口与 nn.MultiheadAttention/标准自注意力兼容，可一行代码替换原 attention 层，支持 Huggingface/官方Transformer等主流框架。

4. **扩展性好**  
   - 支持多种池化方式（DCT/Attention/Mean）、Toeplitz卷积、频谱记忆、wavelet小波增强等，适应多样化需求。

5. **频域门控机制**  
   - 可学习频率分量权重，实现更灵活的信息筛选与融合。

6. **兼容位置编码与padding mask**  
   - 支持标准Transformer的位置信息和序列mask处理，确保NLP任务兼容性。

---

## 使用场景

- **长文本/文档级NLP任务**  
- **大语言模型（LLM）与生成式模型**  
- **机器翻译、问答、摘要等需长距离依赖的任务**  
- **高效推理场景、移动端/低资源设备上的Transformer应用**  
- **与标准Transformer结构无缝结合、替换Attention模块**

---

## 使用方法

### 1. 安装依赖

- 可选：如需启用DCT池化，建议安装 `torch_dct`（否则自动降级为mean pooling）。

### 2. 引入并替换 Attention 层

假设原模型如下：

```python
self.attn = nn.MultiheadAttention(embed_dim, num_heads)
```

替换为 SpectreBlockNLP：

```python
from spectre_nlp import SpectreBlockNLP

self.attn = SpectreBlockNLP(embed_dim=768, num_heads=12)
```

### 3. 前向调用示例

```python
output = self.attn(input_emb, attention_mask=mask, position_embeddings=pos_emb)
```
- `input_emb`: (batch_size, seq_len, embed_dim) 输入embedding
- `attention_mask`: (batch_size, seq_len) 可选，padding位置为0
- `position_embeddings`: (batch_size, seq_len, embed_dim) 可选，绝对或相对位置编码

### 4. 参数说明

可自定义以下参数，满足不同场景需求：

- `embed_dim`: 输入/输出维度
- `num_heads`: 头数
- `n_fft`: FFT长度（建议与最长序列长度接近，如128/256/512）
- `mlp_ratio`: MLP扩展因子
- `d_gate`: 门控MLP隐藏层
- `use_toeplitz`: 是否启用Toeplitz卷积
- `dropout_p`: dropout概率
- `pooling_type`: 池化方式（"dct"/"attention"/"mean"）
- `num_groups`: 分组数
- `num_buckets`: 频率桶数量
- `wavelet_on_rate`: wavelet增强概率
- `memory_size`: 频谱记忆大小

---

## 技术原理简述

SpectreBlockNLP 利用 FFT 将输入序列映射到频域，在频域通过门控MLP、Toeplitz卷积和可选的wavelet小波进行高效混合，然后再逆变换回时域。通过频域混合操作可高效捕捉长距离特征，极大提升 Transformer 在长序列NLP任务中的表现。

---

## 参考文献

- SPECTRE: Frequency-Domain Token Mixing for Transformers
- **The FFT Strikes Back: An Efficient Alternative to Self-Attention**  
  - arXiv: [2502.18394v1](https://arxiv.org/html/2502.18394v1)  
  - [ResearchGate PDF](https://www.researchgate.net/profile/Jacob-Fein-Ashley/publication/389351112_The_FFT_Strikes_Back_An_Efficient_Alternative_to_Self-Attention/links/6812b090bfbe974b23bf6858/The-FFT-Strikes-Back-An-Efficient-Alternative-to-Self-Attention.pdf)
  - [Huggingface论文索引](https://huggingface.co/papers/2502.18394)

---

## 作者信息与版权

作者：Jacob Fein-Ashley 等  
原论文团队：SPECTRE 论文作者组  
代码整理与适配：yunsicjh

版权声明：  
本模块及文档遵循开放源代码协议，欢迎学术研究、个人学习与非商业用途自由使用、修改与分发。  
如需商业用途、二次开发或引用，请注明原作者与论文出处，并遵循相关许可证规定。