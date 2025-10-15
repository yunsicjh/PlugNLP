import torch
import torch.nn as nn


class LinearAttentionNLP(nn.Module):
    """
    适用于NLP任务的线性注意力模块。
    输入: (B, N, C)
    输出: (B, N, C)
    Args:
        dim (int): 输入隐藏维度
        num_heads (int): 注意力头数
        qkv_bias (bool): 是否为qkv投影添加bias
        attn_drop (float): 注意力dropout
        proj_drop (float): 输出dropout
        pos_enc (bool): 是否使用可学习的位置编码
        max_len (int): 最大序列长度
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        pos_enc=True,
        max_len=512,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim 必须能整除 num_heads"
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pos_enc = pos_enc
        self.max_len = max_len
        if pos_enc:
            self.positional_encoding = nn.Parameter(
                torch.zeros(1, max_len, dim)
            )  # 最大长度 max_len

        self.kernel_fn = nn.ReLU()

    def forward(self, x, mask=None):
        """
        x: (B, N, C)
        mask: (B, N) 或 None（可选）
        """
        B, N, C = x.shape
        if self.pos_enc:
            if N > self.max_len:
                # 若输入长度超过最大编码长度，需扩展或截断
                pos_emb = nn.functional.interpolate(
                    self.positional_encoding.transpose(1, 2), size=N, mode="linear"
                ).transpose(1, 2)
            else:
                pos_emb = self.positional_encoding[:, :N, :]
            x = x + pos_emb

        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)  # (B, N, 3, h, c)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, h, N, c)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各为 (B, h, N, c)

        q = self.kernel_fn(q)
        k = self.kernel_fn(k)

        # 归一化项
        k_sum = k.sum(dim=2)  # (B, h, c)
        z = 1 / (torch.einsum("b h n c, b h c -> b h n", q, k_sum) + 1e-6)  # (B, h, N)

        # 主路径
        kv = torch.einsum("b h m c, b h m d -> b h c d", k, v)  # (B, h, c, d)
        out = torch.einsum("b h n c, b h c d -> b h n d", q, kv)  # (B, h, N, d)
        out = out * z.unsqueeze(-1)  # (B, h, N, d)

        # 合并多头
        out = (
            out.permute(0, 2, 1, 3)
            .contiguous()
            .view(B, N, self.num_heads * self.head_dim)
        )  # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


# 示例用法
if __name__ == "__main__":
    attn = LinearAttentionNLP(dim=768, num_heads=12, pos_enc=True, max_len=256)
    x = torch.randn(4, 128, 768)  # (batch, seq_len, hidden_dim)
    out = attn(x)
    print(out.shape)  # (4, 128, 768)
