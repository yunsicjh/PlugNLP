import torch
import math
import torch.nn as nn

"""
创新1：引入多尺度特征融合
问题：当前的CGLU模块仅通过单尺度的3×3深度卷积捕获局部特征，可能对输入特征的多尺度信息感知不足。
改进思路：
1. 引入一个多尺度机制，例如同时添加不同卷积核尺寸（如3×3和5×5）的深度卷积。
2. 不同尺度的特征可以在深度卷积后通过拼接或加权融合实现增强的特征表达。

创新2：引入通道自适应机制
问题：当前的CGLU模块中，通道间的权重调整是隐式的（通过GLU中的门控分支实现），缺乏全局上下文信息的显式
建模能力。
改进思路：
添加一个轻量级的通道注意力机制（如Squeeze-and-Excitation, SE）到CGLU模块中，通过显式建模全局上下文，
对特征进行通道维度的自适应增强。
"""


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim
        )

    def forward(self, x, H, W):
        B, N, C = x.shape

        # 计算需要的padding
        target_len = H * W
        if N < target_len:
            pad_len = target_len - N
            pad = torch.zeros(B, pad_len, C, device=x.device)
            x_padded = torch.cat([x, pad], dim=1)
        else:
            x_padded = x[:, :target_len, :]

        # 重塑和卷积
        x_reshaped = x_padded.transpose(1, 2).view(B, C, H, W)
        x_conv = self.dwconv(x_reshaped)
        x_out = x_conv.flatten(2).transpose(1, 2)

        # 如果之前做了padding，这里要去掉
        if N < target_len:
            x_out = x_out[:, :N, :]

        return x_out


class CGLU(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H=None, W=None):
        # 自动计算合适的H和W
        B, N, C = x.shape
        if H is None or W is None:
            H = max(3, int(math.ceil(math.sqrt(N))))  # 至少为3以适应3x3卷积
            W = H

        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 测试代码
if __name__ == "__main__":
    models = CGLU(in_features=768, hidden_features=512, out_features=768)

    print("\n=== NLP短序列测试 ===")
    # 测试短序列
    B, N, C = 2, 8, 768  # 较短的序列长度
    input_short = torch.randn(B, N, C)
    output_short = models(input_short)  # H,W会自动设置
    print("Short sequence input size:", input_short.size())
    print("Short sequence output size:", output_short.size())

    print("\n=== NLP长序列测试 ===")
    # 测试长序列
    B, N, C = 2, 196, 768
    H, W = 14, 14
    input_long = torch.randn(B, N, C)
    output_long = models(input_long, H, W)
    print("Long sequence input size:", input_long.size())
    print("Long sequence output size:", output_long.size())
