# Gated Convolution Linear Units (GCLU)

## 1. 论文信息

- **标题**: TransNeXt: Robust Foveal Visual Perception for Vision Transformers
- **发表**: CVPR 2024
- **arXiv**: [2311.17132](https://arxiv.org/abs/2311.17132)
- **DOI**: [10.48550/arXiv.2311.17132](https://doi.org/10.48550/arXiv.2311.17132)
- **领域**: Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI)

## 2. 模块优势

1. **局部-全局感知能力**:
   - DWConv 提供局部空间建模
   - GLU 门控机制实现全局特征交互
   - 自适应感受野动态调节

2. **计算效率**:
   - 深度可分离卷积降低参数量
   - 序列自适应填充避免冗余计算
   - 门控机制提供特征选择性激活

3. **灵活性与鲁棒性**:
   - 支持动态序列长度
   - 自动计算空间维度 (H,W)
   - 处理不规则序列的填充策略

4. **NLP友好设计**:
   - 保持序列特征表示 (B,N,C)
   - 支持短序列和长序列处理
   - 无需预定义空间结构

## 3. 适用任务

### 3.1 序列建模任务
- 长文本理解
- 代码分析
- 时序预测
- 文本分类
- 序列标注

### 3.2 多模态任务
- 视觉-语言预训练
- 图文匹配
- 跨模态检索
- 多模态融合

### 3.3 特征增强
- Transformer 网络增强
- 特征提取管道
- 注意力机制补充
- 表示学习

## 4. 使用示例

### 4.1 短序列处理

```python
import torch
from gclu_gate.gclu_nlp import CGLU

# 初始化模块
model = CGLU(
    in_features=768,      # 输入特征维度
    hidden_features=512,  # 隐藏层维度(会被自动调整为 2/3)
    out_features=768,     # 输出特征维度
    act_layer=nn.GELU,    # 激活函数
    drop=0.1             # Dropout 率
)

# 处理短序列
B, N, C = 2, 8, 768     # 批次, 序列长度, 通道数
x = torch.randn(B, N, C)
output = model(x)       # H,W 自动计算
print(output.shape)     # torch.Size([2, 8, 768])
```

### 4.2 长序列处理

```python
# 处理长序列(指定 H,W)
B, N, C = 2, 196, 768
H, W = 14, 14          # 可选: 指定空间维度
x = torch.randn(B, N, C)
output = model(x, H, W)
print(output.shape)    # torch.Size([2, 196, 768])
```

## 5. 实现细节

### 5.1 深度可分离卷积 (DWConv)

```python
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, 
            kernel_size=3,
            stride=1, 
            padding=1,
            groups=dim  # 分组卷积
        )
```

特点:
- 3x3 卷积核捕获局部结构
- 分组卷积减少参数量
- 保持特征维度不变

### 5.2 序列填充策略

```python
# 计算需要的padding
target_len = H * W
if N < target_len:
    pad_len = target_len - N
    pad = torch.zeros(B, pad_len, C)
    x_padded = torch.cat([x, pad], dim=1)
```

优势:
- 自适应序列长度
- 不损失原始信息
- 支持不规则序列

### 5.3 GLU 门控机制

```python
# 特征分割与门控
x, v = self.fc1(x).chunk(2, dim=-1)
x = self.act(self.dwconv(x, H, W)) * v
```

作用:
- 动态特征选择
- 非线性变换
- 梯度流控制

## 6. 注意事项

1. **维度设置**:
   - 输入特征维度需与模型配置匹配
   - hidden_features 会被自动调整为 2/3
   - H,W 可选参数影响感受野

2. **序列长度**:
   - 自动填充处理不规则序列
   - 建议长序列指定 H,W 参数
   - 最小序列长度为9(3x3)

3. **内存优化**:
   - 适当设置 dropout 防止过拟合
   - 注意批次大小与序列长度平衡
   - 长序列考虑梯度累积

## 7. 依赖要求

```text
torch>=1.8.0
```

## 8. License

MIT License

## 9. 文档信息

- **作者**: yunsicjh
- **更新时间**: 2025-10-14
- **仓库**: [PlugNLP](https://github.com/yunsicjh/PlugNLP)
- **模块**: Gated Convolution Linear Units (GCLU)
- **维护**: 如有问题请提交 Issue 或 PR
