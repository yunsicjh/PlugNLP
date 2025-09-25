import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class StandardAttentionWithRoPE(nn.Module):
    """标准 Multi-Head Attention 集成 Rotary Position Embedding (RoPE)。

    实现特性:
    - 支持任意 batch 内相同长度序列的自注意力（可扩展跨注意力用法）
    - RoPE 采用 pairwise (even, odd) 旋转实现，与 RoFormer 一致
    - 可选返回注意力权重 (return_attn=True)

    参数:
        embed_dim: hidden size
        num_heads: multi-head 数量
        dropout: attention 权重 dropout 概率
    输入张量形状:
        query/key/value: (B, N, D)
        rope_embeddings: (cos, sin) 其中 cos/sin: (N_rope, D_head/2)
        attn_mask: 可选 (B, 1, 1, N) 或 (B, 1, N, N) 或 broadcast 到 (B, H, N, N)
    输出:
        y: (B, N, D)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 2 == 0, "RoPE 需要 head_dim 为偶数 (方便偶/奇配对)"
        self.dropout_layer = nn.Dropout(dropout)

        # Q,K,V 投影 & 输出层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    # ---------------- Public API ----------------
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rope_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        B, Nq, _ = query.shape
        Nk = key.shape[1]
        assert key.shape == value.shape, "key 与 value 形状需一致 (B, Nk, D)"

        # 线性映射 -> (B, N, H, Dh)
        q = self.q_proj(query).view(B, Nq, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(B, Nk, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(B, Nk, self.num_heads, self.head_dim)

        if rope_embeddings is not None:
            cos, sin = rope_embeddings
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

        # (B,H,N,Dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out, attn_weights = self._scaled_dot_product_attention(q, k, v, attn_mask)

        # 回到 (B,N,D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Nq, self.embed_dim)
        out = self.out_proj(attn_out)
        return (out, attn_weights) if return_attn else out

    # ---------------- RoPE Core ----------------
    def _apply_rope(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """对输入张量应用 RoPE。

        x: (B, N, H, Dh)
        cos/sin: (N_rope, Dh/2)
        返回: 同形状张量
        """
        B, N, H, Dh = x.shape
        half = Dh // 2
        # 验证 cos/sin 维度
        if cos.shape[-1] != half:
            raise ValueError(
                f"cos/sin 的最后一维应为 Dh/2={half}，但得到 {cos.shape[-1]}"
            )
        # 若 RoPE 长度不足，补最后一行；若更长，裁剪
        if cos.shape[0] < N:
            pad_len = N - cos.shape[0]
            cos = torch.cat([cos, cos[-1:].expand(pad_len, -1)], dim=0)
            sin = torch.cat([sin, sin[-1:].expand(pad_len, -1)], dim=0)
        elif cos.shape[0] > N:
            cos = cos[:N]
            sin = sin[:N]

        # 重塑为复数形式: (B, N, H, half, 2)
        x_pair = x.view(B, N, H, half, 2)
        x_even = x_pair[..., 0]
        x_odd = x_pair[..., 1]

        # broadcast: (1,N,1,half)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        out = torch.stack([rot_even, rot_odd], dim=-1).view(B, N, H, Dh)
        return out

    # ---------------- Attention Core ----------------
    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        Dh = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        if attn_mask is not None:
            # 允许多种形状广播
            scores = scores + attn_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        out = torch.matmul(attn_weights, v)
        return out, attn_weights


def generate_rope_embeddings(
    seq_len: int,
    head_dim: int,
    base: int = 10000,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """生成 RoPE 所需 cos/sin。

    head_dim: 每头维度 (必须为偶数)。
    返回: (cos, sin) 形状 (seq_len, head_dim/2)
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim 必须为偶数以支持 RoPE")
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


# ---------------- Minimal Self-Test ----------------
if __name__ == "__main__":
    attn = StandardAttentionWithRoPE(embed_dim=512, num_heads=8, dropout=0.1)
    B, N = 2, 128
    x = torch.randn(B, N, 512)
    cos, sin = generate_rope_embeddings(N, 512 // 8, device=x.device, dtype=x.dtype)
    y, w = attn(x, x, x, rope_embeddings=(cos, sin), return_attn=True)
    print("Input:", x.shape, "Output:", y.shape, "Attn:", w.shape)
