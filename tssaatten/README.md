# Token Statistics Self-Attention (TSSA)

## 1. 论文信息

- **标题**: Token Statistics Transformer: Linear-Time Attention via Variational Rate Reduction
- **发表**: ICLR 2025
- **arXiv**: [2412.17810](https://arxiv.org/abs/2412.17810)
- **DOI**: [10.48550/arXiv.2412.17810](https://doi.org/10.48550/arXiv.2412.17810)
- **页数**: 24 pages, 11 figures
- **领域**: Machine Learning (cs.LG)

## 2. 模块优势

1. **线性时间复杂度**: 
   - 传统自注意力: O(n²)，n 为序列长度
   - TSSA: O(n)，通过变分率降低实现线性扩展
   - 显著降低长序列处理的内存和计算开销

2. **理论基础扎实**:
   - 基于信息论的变分率降低框架
   - 通过统计势能(Statistical Potential)建模 token 间相互作用
   - 保证注意力权重的概率解释性

3. **灵活适配性**:
   - 支持 CV (图像) 和 NLP (文本) 两类任务
   - 无需预训练即可快速部署
   - 保持 transformer 架构的通用性

4. **资源效率**:
   - 参数量小于标准自注意力
   - 训练和推理显存占用低
   - 适合部署在资源受限设备

## 3. 适用任务

### 3.1 NLP 任务
- 长文本处理与理解
- 文档摘要生成
- 机器翻译
- 代码补全
- 对话系统

### 3.2 CV 任务
- 图像分类
- 目标检测
- 语义分割
- 图像生成
- 视频理解

### 3.3 其他潜在应用
- 多模态融合
- 时序数据分析
- 图结构数据处理

## 4. 使用示例

### 4.1 NLP 任务用法

```python
import torch
from tssaatten.tssaatten_nlp import TSSA

# 初始化模块
dim = 64           # 输入特征维度
num_heads = 8      # 注意力头数
model = TSSA(
    dim=dim,
    num_heads=num_heads,
    qkv_bias=False,    # 是否使用偏置
    attn_drop=0.1,     # 注意力dropout率
    proj_drop=0.1      # 投影层dropout率
)

# 准备输入: [batch_size, seq_len, dim]
B, N, C = 1, 1024, 64  # 批次, 序列长度, 特征维度
x = torch.randn(B, N, C)

# 前向计算
output = model(x)  # 输出形状: [1, 1024, 64]
```

### 4.2 CV 任务用法

```python
import torch
from tssaatten.tssaatten_nlp import TSSA

# 初始化模块
model = TSSA(dim=64, num_heads=8)

# 准备输入: [B, C, H, W]
x = torch.randn(1, 64, 32, 32)

# 重排序处理
x = x.reshape(1, 64, -1).transpose(-1, -2)  # [B, H*W, C]

# 前向计算
output = model(x)

# 恢复空间维度
output = output.view(1, 64, 32, 32)  # [B, C, H, W]
```

## 5. 实现细节

### 5.1 关键组件

1. **自适应温度参数**:
```python
self.temp = nn.Parameter(torch.ones(num_heads, 1))
```
- 每个注意力头独立学习温度缩放因子
- 动态调节注意力分布的锐度

2. **统计势计算**:
```python
w_normed = torch.nn.functional.normalize(w, dim=-2)
w_sq = w_normed**2
Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)
```
- 计算归一化统计势
- 通过温度调节统计势强度

3. **注意力权重**:
```python
attn = 1.0 / (1 + dots)
```
- 基于变分率原理的反比例注意力机制
- 避免 softmax 运算提升效率

## 6. 注意事项

1. **维度对齐**:
   - 输入特征维度必须能被注意力头数整除
   - CV 任务需要正确处理空间维度重排

2. **数值稳定性**:
   - 统计势计算中包含防止除零保护 (1e-8)
   - 注意力归一化采用数值稳定实现

3. **内存效率**:
   - 适当选择 batch size 和序列长度
   - 根据实际需求调整头数和特征维度

## 7. License

MIT License

## 8. 文档信息

- **作者**: yunsicjh
- **更新时间**: 2025-10-14
- **仓库**: [PlugNLP](https://github.com/yunsicjh/PlugNLP)
- **模块**: Token Statistics Self-Attention (TSSA)
- **维护**: 如有问题请提交 Issue 或 PR
