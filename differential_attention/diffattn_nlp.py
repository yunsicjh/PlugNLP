"""
Unified Differential Attention Module - 即插即用的差分注意力模块
整合了高效RoPE位置编码、RMSNorm标准化和差分注意力机制
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

# Triton加速（如果可用）
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ============================================================================
# RMSNorm Implementation
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine=True,
        memory_efficient=False,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


# ============================================================================
# RoPE Implementation (Triton accelerated)
# ============================================================================
if TRITON_AVAILABLE:

    @triton.jit
    def rotary_kernel(
        OUT,
        X,
        COS,
        SIN,
        CU_SEQLENS,
        SEQLEN_OFFSETS,
        seqlen,
        nheads,
        rotary_dim,
        seqlen_ro,
        CACHE_KEY_SEQLEN,
        stride_out_batch,
        stride_out_seqlen,
        stride_out_nheads,
        stride_out_headdim,
        stride_x_batch,
        stride_x_seqlen,
        stride_x_nheads,
        stride_x_headdim,
        BLOCK_K: tl.constexpr,
        IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        CONJUGATE: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_batch = tl.program_id(axis=1)
        pid_head = tl.program_id(axis=2)
        rotary_dim_half = rotary_dim // 2

        if not IS_VARLEN:
            X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
            OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
        else:
            start_idx = tl.load(CU_SEQLENS + pid_batch)
            seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
            X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
            OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

        if pid_m * BLOCK_M >= seqlen:
            return
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        if not IS_SEQLEN_OFFSETS_TENSOR:
            rm_cs = rm + SEQLEN_OFFSETS
        else:
            rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
        rk = tl.arange(0, BLOCK_K)
        rk_half = tl.arange(0, BLOCK_K // 2)

        if not INTERLEAVED:
            X = X + (
                rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim
            )
            COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
            SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
            cos = tl.load(
                COS,
                mask=(rm_cs[:, None] < seqlen_ro)
                & (rk_half[None, :] < rotary_dim_half),
                other=1.0,
            ).to(tl.float32)
            sin = tl.load(
                SIN,
                mask=(rm_cs[:, None] < seqlen_ro)
                & (rk_half[None, :] < rotary_dim_half),
                other=0.0,
            ).to(tl.float32)
            x0 = tl.load(
                X,
                mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
                other=0.0,
            ).to(tl.float32)
            x1 = tl.load(
                X + rotary_dim_half * stride_x_headdim,
                mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
                other=0.0,
            ).to(tl.float32)
            if CONJUGATE:
                sin = -sin
            o0 = x0 * cos - x1 * sin
            o1 = x0 * sin + x1 * cos
            OUT = OUT + (
                rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim
            )
            tl.store(
                OUT,
                o0,
                mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            )
            tl.store(
                OUT + rotary_dim_half * stride_out_headdim,
                o1,
                mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            )
        else:
            rk_swap = rk + ((rk + 1) % 2) * 2 - 1
            rk_repeat = tl.arange(0, BLOCK_K) // 2
            X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
            X1 = X + (
                rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim
            )
            COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
            SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
            cos = tl.load(
                COS,
                mask=(rm_cs[:, None] < seqlen_ro)
                & (rk_repeat[None, :] < rotary_dim_half),
                other=1.0,
            ).to(tl.float32)
            sin = tl.load(
                SIN,
                mask=(rm_cs[:, None] < seqlen_ro)
                & (rk_repeat[None, :] < rotary_dim_half),
                other=0.0,
            ).to(tl.float32)
            x0 = tl.load(
                X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0
            ).to(tl.float32)
            x1 = tl.load(
                X1,
                mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim),
                other=0.0,
            ).to(tl.float32)
            if CONJUGATE:
                sin = -sin
            x0_cos = x0 * cos
            x1_sin = x1 * sin
            out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
            OUT = OUT + (
                rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim
            )
            tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        # Fallback to pure PyTorch implementation
        return _apply_rotary_pytorch(x, cos, sin, interleaved=interleaved)

    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert (
            max_seqlen is not None
        ), "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = (
        32
        if rotary_dim <= 32
        else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    )
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)

    with torch.cuda.device(x.device.index):
        rotary_kernel[grid](
            output,
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,
            nheads,
            rotary_dim,
            seqlen_ro,
            seqlen // 128,
            output.stride(0) if not is_varlen else 0,
            output.stride(-3),
            output.stride(-2),
            output.stride(-1),
            x.stride(0) if not is_varlen else 0,
            x.stride(-3),
            x.stride(-2),
            x.stride(-1),
            BLOCK_K,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M,
        )
    return output


def _apply_rotary_pytorch(x, cos, sin, interleaved=False):
    """PyTorch fallback implementation"""
    batch, seq_len, nheads, head_dim = x.shape

    # Ensure cos/sin have correct shape
    if cos.size(0) < seq_len:
        # Extend cos/sin if needed
        extra_len = seq_len - cos.size(0)
        cos_extra = cos[-1:].expand(extra_len, -1)
        sin_extra = sin[-1:].expand(extra_len, -1)
        cos = torch.cat([cos, cos_extra], dim=0)
        sin = torch.cat([sin, sin_extra], dim=0)

    # Reshape cos/sin to match x dimensions: (1, seq_len, 1, rotary_dim//2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)

    if interleaved:
        # For interleaved RoPE: [x0, x1, x2, x3, ...] -> [x0*cos-x1*sin, x0*sin+x1*cos, x2*cos-x3*sin, ...]
        x_reshaped = x.view(batch, seq_len, nheads, -1, 2)  # (..., dim//2, 2)
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        output = torch.stack([o1, o2], dim=-1).view(batch, seq_len, nheads, head_dim)
    else:
        # For traditional RoPE: split into two halves
        d = head_dim // 2
        x1, x2 = x[..., :d], x[..., d:]
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        output = torch.cat([o1, o2], dim=-1)
    return output


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets=0,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        out = apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = apply_rotary(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets=0,
    cu_seqlens=None,
    max_seqlen=None,
):
    return ApplyRotaryEmb.apply(
        x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
    )


# ============================================================================
# Utility Functions
# ============================================================================
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


# ============================================================================
# Differential Attention Implementation
# ============================================================================
class MultiheadDiffAttn(nn.Module):
    """
    差分多头注意力机制，适用于NLP任务
    整合了RoPE位置编码和高效实现
    """

    def __init__(self, embed_dim, depth, num_heads, num_kv_heads=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Lambda parameters for differential attention
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x, rel_pos, attn_mask=None):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            rel_pos: tuple of (cos, sin) for rotary position encoding
            attn_mask: optional attention mask

        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        # Apply RoPE
        q = apply_rotary_emb(q, *rel_pos, interleaved=True).clone()
        k = apply_rotary_emb(k, *rel_pos, interleaved=True).clone()

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))

        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        # Apply differential attention
        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(
            bsz, tgt_len, self.num_heads * 2 * self.head_dim
        )

        attn = self.out_proj(attn)
        return attn


# ============================================================================
# Helper function to generate RoPE embeddings
# ============================================================================
def generate_rope_embeddings(
    seq_len, head_dim, device="cpu", dtype=torch.float32, base=10000
):
    """Generate RoPE cos/sin embeddings"""
    dim = head_dim
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)
    )
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin
