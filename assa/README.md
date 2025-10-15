# Adaptive Sparse Self-Attention (ASSA)

## 1. 论文信息

- **标题**: Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement
- **发表**: CVPR 2024
- **领域**: Computer Vision and Pattern Recognition

## 2. NLP 任务优势

1. **自适应稀疏注意力**:
   - 动态学习注意力权重分布
   - 自适应稀疏化降低计算复杂度
   - 保持关键信息流动性

2. **窗口化局部建模**:
   - 灵活的窗口大小设置（默认 8x8）
   - 支持移位窗口机制（shift_size）
   - 高效处理长序列依赖

3. **特征细化机制**:
   - LeFF (Local enhanced Feed-Forward) 或 FRFN (Feature Refinement Feed-forward Network)
   - 结合局部感受野与全局信息
   - 增强特征表示能力

4. **多头结构优化**:
   - 高效的线性投影实现
   - 相对位置编码增强
   - 可学习的位置偏置

## 3. 适用任务

### 3.1 文本处理
- 长文档理解
- 文本分类
- 序列标注
- 文本生成
- 机器翻译

### 3.2 跨模态任务
- 视觉问答
- 图文匹配
- 多模态融合
- 文档布局分析

### 3.3 序列建模
- 时序预测
- 事件序列分析
- 用户行为建模
- 异常检测

## 4. 使用示例

### 4.1 NLP 任务用法

```python
import torch
from assa.assa_nlp import ASSA

# 初始化模块
model = ASSA(
    dim=64,                    # 特征维度
    input_resolution=(32, 32), # 输入分辨率
    num_heads=8,              # 注意力头数
    win_size=8,              # 窗口大小
    shift_size=0,            # 移位大小
    mlp_ratio=4.0,          # FFN 隐层比例
    token_mlp="leff"        # 'leff', 'frfn' 或 'ffn'
)

# 准备输入: [B, L, N]
B, L, N = 1, 1024, 64  # 批次, 序列长度, 特征维度
x = torch.randn(B, L, N)

# 前向计算
output = model(x)  # 输出形状同输入
```

### 4.2 跨模态处理示例

```python
# 对于图像特征序列化处理
H, W = 32, 32
x = torch.randn(1, 64, H, W)  # [B, C, H, W]

# 转换为序列形式
x = rearrange(x, 'b c h w -> b (h w) c')

# ASSA 处理
output = model(x)

# 恢复空间维度（如需）
output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
```

## 5. 实现细节

### 5.1 核心组件

1. **自适应窗口注意力**:
```python
class WindowAttention_sparse(nn.Module):
    def __init__(self, dim, win_size, num_heads...):
        # 自适应权重
        self.w = nn.Parameter(torch.ones(2))
```

2. **位置编码增强**:
```python
# 相对位置偏置
self.relative_position_bias_table = nn.Parameter(
    torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads)
)
```

3. **特征细化网络**:
```python
class FRFN(nn.Module):
    # 特征细化前馈网络
    def __init__(self, dim, hidden_dim...):
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv, self.dim_conv, 3, 1, 1, bias=False
        )
```

## 6. 注意事项

1. **维度设置**:
   - 输入维度必须与配置匹配
   - 序列长度需为完全平方数
   - 窗口大小应小于输入分辨率

2. **内存效率**:
   - 大序列推荐使用稀疏注意力
   - 适当调整窗口大小平衡性能
   - 注意力掩码自动处理

3. **模型选择**:
   - LeFF 适合轻量级任务
   - FRFN 适合复杂特征提取
   - 可配置 sparse 开关

## 7. 依赖要求

```text
torch>=1.8.0
timm
einops
```

## 8. License

MIT License

## 9. 文档信息

- **作者**: yunsicjh
- **更新时间**: 2025-10-14
- **仓库**: [PlugNLP](https://github.com/yunsicjh/PlugNLP)
- **模块**: Adaptive Sparse Self-Attention (ASSA)
- **维护**: 如有问题请提交 Issue 或 PR