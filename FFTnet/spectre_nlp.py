import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =================== 频域辅助层 ===================
class ComplexModReLU(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))
        nn.init.constant_(self.bias, -0.1)
        self.register_buffer("eps", torch.tensor(1e-4))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(z)
        mag_stable = torch.sqrt(mag.square() + self.eps.square())
        scale = F.relu(mag + self.bias) / mag_stable
        return z * scale


class DCTPooling(nn.Module):
    def __init__(self, embed_dim: int, dct_components: int = 64):
        super().__init__()
        self.dct_components = dct_components
        self.embed_dim = embed_dim
        try:
            import torch_dct as dct

            self.dct = dct
            self.has_dct = True
        except ImportError:
            self.has_dct = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_dct:
            x_dct = self.dct.dct(x.transpose(1, 2))
            x_pool = x_dct[:, :, : self.dct_components].mean(dim=2)
        else:
            x_pool = x.mean(dim=1)
        return x_pool


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.w2(self.activation(self.w1(x)))
        weights = F.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1)
        return pooled


class MeanPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


# ========== 小波变换实现（Haar DWT/IDWT） ==========
class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__()
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        self.register_buffer("h0_base", torch.tensor([sqrt2_inv, sqrt2_inv]))
        self.register_buffer("h1_base", torch.tensor([-sqrt2_inv, sqrt2_inv]))

    def forward(self, x, levels=1):
        if x.ndim == 2:
            x = x[:, None, :]
        B, C, L = x.shape
        low, high = [], []
        h0 = self.h0_base.repeat(C).view(C, 1, 2)
        h1 = self.h1_base.repeat(C).view(C, 1, 2)
        for _ in range(levels):
            x_padded = F.pad(x, (1, 0), mode="circular")
            current_L = x.shape[-1]
            lo = F.conv1d(x_padded, h0, stride=2, groups=C)
            hi = F.conv1d(x_padded, h1, stride=2, groups=C)
            if lo.shape[-1] * 2 > current_L:
                lo = lo[..., :-1]
                hi = hi[..., :-1]
            low.append(lo)
            high.append(hi)
            x = lo
        return low, high


class HaarIDWT(nn.Module):
    def __init__(self):
        super().__init__()
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        self.register_buffer("g0_base", torch.tensor([sqrt2_inv, sqrt2_inv]))
        self.register_buffer("g1_base", torch.tensor([sqrt2_inv, -sqrt2_inv]))

    def forward(self, low, high):
        x = low[-1]
        for i in range(len(high) - 1, -1, -1):
            B, C, L = x.shape
            lo_coeff = x
            hi_coeff = high[i]
            g0_filter = self.g0_base.repeat(C).view(C, 1, 2)
            g1_filter = self.g1_base.repeat(C).view(C, 1, 2)
            lo_up = F.conv_transpose1d(lo_coeff, g0_filter, stride=2, groups=C)
            hi_up = F.conv_transpose1d(hi_coeff, g1_filter, stride=2, groups=C)
            target_length = L * 2
            if lo_up.shape[-1] > target_length:
                lo_up = lo_up[..., :-1]
                hi_up = hi_up[..., :-1]
            x = lo_up + hi_up
        return x


_haar_dwt = None


def dwt1d_harr(x, levels=1):
    global _haar_dwt
    if _haar_dwt is None:
        _haar_dwt = HaarDWT()
    if _haar_dwt.h0_base.device != x.device:
        _haar_dwt = _haar_dwt.to(x.device)
    return _haar_dwt(x, levels)


_haar_idwt = None


def idwt1d_harr(low, high):
    global _haar_idwt
    if _haar_idwt is None:
        _haar_idwt = HaarIDWT()
    device = low[0].device if low else high[0].device
    if _haar_idwt.g0_base.device != device:
        _haar_idwt = _haar_idwt.to(device)
    return _haar_idwt(low, high)


def dwt_decompose(x, levels=None):
    if x.ndim == 2:
        x = x.unsqueeze(1)
    B, C, L = x.shape
    if levels is None:
        levels = int(math.log2(L))
    coeffs = []
    for level in range(levels):
        low, high = dwt1d_harr(x, levels=1)
        coeffs.append(high[0])
        x = low[0]
        if x.shape[-1] <= 1:
            break
    coeffs.append(x)
    return coeffs


def dwt_reconstruct(coeffs):
    x = coeffs[-1]
    for i in range(len(coeffs) - 2, -1, -1):
        low = [x]
        high = [coeffs[i]]
        x = idwt1d_harr(low, high)
    return x.squeeze(1) if x.shape[1] == 1 else x


# ============= 复数插值与卷积辅助 ==============
def interp_complex_1d(x: torch.Tensor, size: int, mode: str = "linear") -> torch.Tensor:
    B, G, K = x.shape
    device = x.device
    _PYTORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
    _USE_GRID_SAMPLE_CUBIC = _PYTORCH_VERSION >= (2, 2)
    if mode == "cubic":
        if _USE_GRID_SAMPLE_CUBIC:
            real_imag = torch.stack([x.real, x.imag], dim=1).reshape(B * G, 2, 1, K)
            grid_x = torch.linspace(-1, 1, size, device=device)
            grid = grid_x.view(1, 1, size, 1).expand(B * G, 1, size, 1)
            grid_2d = torch.cat([grid, torch.zeros_like(grid)], dim=-1)
            interp = F.grid_sample(
                real_imag,
                grid_2d,
                mode="bicubic",
                padding_mode="border",
                align_corners=True,
            )
            real_up = interp[:, 0, 0, :]
            imag_up = interp[:, 1, 0, :]
            up = torch.complex(real_up, imag_up).view(B, G, size)
            return up
        else:
            mode = "linear"
    assert mode in ("linear", "nearest"), f"Unsupported interpolation mode: {mode}"
    real = x.real.reshape(B * G, 1, K)
    imag = x.imag.reshape(B * G, 1, K)
    if mode == "linear":
        real_up = F.interpolate(real, size=size, mode=mode, align_corners=True)
        imag_up = F.interpolate(imag, size=size, mode=mode, align_corners=True)
    else:
        real_up = F.interpolate(real, size=size, mode=mode)
        imag_up = F.interpolate(imag, size=size, mode=mode)
    up = torch.view_as_complex(
        torch.stack([real_up.squeeze(1), imag_up.squeeze(1)], dim=-1)
    )
    return up.view(B, G, size)


def complex_conv1d(x: torch.Tensor, kernel: torch.Tensor, padding: int) -> torch.Tensor:
    x_r, x_i = x.real, x.imag
    k_r, k_i = kernel.real, kernel.imag
    *batch_shape, L = x.shape
    batch_size = math.prod(batch_shape) if batch_shape else 1
    x_r_flat = x_r.reshape(batch_size, 1, L)
    x_i_flat = x_i.reshape(batch_size, 1, L)
    k_r_view = k_r.view(1, 1, -1)
    k_i_view = k_i.view(1, 1, -1)
    try:
        conv_ac = F.conv1d(x_r_flat, k_r_view, padding=padding, padding_mode="circular")
        conv_bd = F.conv1d(x_i_flat, k_i_view, padding=padding, padding_mode="circular")
        conv_ad = F.conv1d(x_r_flat, k_i_view, padding=padding, padding_mode="circular")
        conv_bc = F.conv1d(x_i_flat, k_r_view, padding=padding, padding_mode="circular")
    except Exception:
        x_r_pad = F.pad(x_r_flat, (padding, padding), mode="circular")
        x_i_pad = F.pad(x_i_flat, (padding, padding), mode="circular")
        conv_ac = F.conv1d(x_r_pad, k_r_view, padding=0)
        conv_bd = F.conv1d(x_i_pad, k_i_view, padding=0)
        conv_ad = F.conv1d(x_r_pad, k_i_view, padding=0)
        conv_bc = F.conv1d(x_i_pad, k_r_view, padding=0)
    real_part = (conv_ac - conv_bd).reshape(*batch_shape, L)
    imag_part = (conv_ad + conv_bc).reshape(*batch_shape, L)
    return torch.complex(real_part, imag_part)


def pruned_irfft_single(X_half: torch.Tensor, n: int, pos: int) -> torch.Tensor:
    F_half, d = X_half.shape
    k = torch.arange(F_half, device=X_half.device, dtype=X_half.real.dtype)
    phase = 2 * math.pi * k * pos / n
    cos_phase = torch.cos(phase).unsqueeze(1)
    sin_phase = torch.sin(phase).unsqueeze(1)
    contrib = X_half.real * cos_phase - X_half.imag * sin_phase
    result = contrib[0]
    if n % 2 == 0:
        result += 2 * contrib[1:-1].sum(dim=0)
        result += contrib[-1] * ((-1) ** pos)
    else:
        result += 2 * contrib[1:].sum(dim=0)
    return result / n


# ================= SPECTRE头 ====================
class SpectreHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        fft_size: int,
        *,
        num_groups: int = 4,
        num_buckets: Optional[int] = None,
        d_gate: int = 256,
        use_toeplitz: bool = False,
        toeplitz_bw: int = 4,
        dropout_p: float = 0.0,
        pooling_type: str = "dct",
    ):
        super().__init__()
        self.d = embed_dim
        self.n_fft = fft_size
        self.G = num_groups
        self.d_g = embed_dim // num_groups
        assert embed_dim % num_groups == 0, "embed_dim must be divisible by num_groups"
        self.F_half = fft_size // 2 + 1
        self.B = max(4, num_buckets or int(math.sqrt(self.F_half)))
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        out_dim = self.B * self.G * 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim, d_gate),
            nn.GELU(),
            nn.Linear(d_gate, out_dim),
        )
        self.q_norm = nn.LayerNorm(embed_dim)
        self.modrelu = ComplexModReLU(self.F_half * self.G)
        if pooling_type == "dct":
            self.pooling = DCTPooling(embed_dim)
        elif pooling_type == "attention":
            self.pooling = AttentionPooling(embed_dim)
        else:
            self.pooling = MeanPool()
        self.use_toeplitz = use_toeplitz
        self.toeplitz_kernel = None
        if use_toeplitz:
            self.toeplitz_bw = toeplitz_bw
            self.register_parameter("toeplitz_kernel", None)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self._reset_parameters()

    def _reset_parameters(self):
        if self.use_toeplitz and self.toeplitz_kernel is None:
            device = self.W_q.weight.device
            dtype = torch.cfloat
            self.toeplitz_kernel = nn.Parameter(
                torch.randn(2 * self.toeplitz_bw + 1, device=device, dtype=dtype)
                / math.sqrt(2 * self.toeplitz_bw + 1)
            )

    def forward(
        self,
        x: torch.Tensor,
        pos_phase: Optional[torch.Tensor] = None,
        return_q_pool: bool = False,
        memory_fft: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, d = x.shape
        assert d == self.d
        Q = self.W_q(x)
        V = self.W_v(x)
        V_fft = torch.fft.rfft(V, n=self.n_fft, dim=1)
        q_pool = self.q_norm(self.pooling(Q))
        Bsz, d_pool = q_pool.shape
        gate_rs = self.gate_mlp(q_pool).view(Bsz, self.G, self.B, 2)
        gate_anchor = torch.view_as_complex(gate_rs)
        if self.use_toeplitz:
            conv_result = complex_conv1d(
                gate_anchor, self.toeplitz_kernel, self.toeplitz_bw
            )
            gate_anchor = gate_anchor + conv_result
        gate_half = interp_complex_1d(gate_anchor, size=self.F_half, mode="cubic")
        gate_half = self.modrelu(gate_half.reshape(Bsz, -1)).view_as(gate_half)
        if pos_phase is not None:
            gate_half = gate_half * pos_phase.unsqueeze(
                1 if pos_phase.dim() == 2 else 0
            )
        gate_broadcast = gate_half.permute(0, 2, 1)
        gate_broadcast = gate_broadcast.repeat_interleave(self.d_g, dim=-1)
        mixed_half = gate_broadcast * V_fft
        if memory_fft is not None:
            mixed_half = mixed_half + memory_fft.unsqueeze(0)
        v_time = torch.fft.irfft(mixed_half, n=self.n_fft, dim=1)
        result = self.dropout(v_time[:, :N])
        if return_q_pool:
            return result, q_pool
        return result


# =========== 多头SPECTRE wrapper ============
class WaveletRefinement(nn.Module):
    def __init__(self, embed_dim: int, on_rate: float = 0.1):
        super().__init__()
        self.on_rate = on_rate
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, v: torch.Tensor, q_pool: torch.Tensor):
        B, N, d = v.shape
        on_mask = torch.rand(B, 1, 1, device=v.device) < self.on_rate
        if not on_mask.any():
            return v
        gate = self.gate_mlp(q_pool).unsqueeze(1)
        outputs = []
        for b in range(B):
            if on_mask[b]:
                v_b = v[b]
                v_b_t = v_b.t().unsqueeze(0)
                coeffs = dwt_decompose(v_b_t)
                v_ref = dwt_reconstruct(coeffs)
                v_ref = v_ref.squeeze(0).t()
                outputs.append(v_ref)
            else:
                outputs.append(v[b])
        v_ref = torch.stack(outputs, dim=0)
        residual = (v_ref.detach() * gate) * on_mask
        return v + residual


class SpectreMultiHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        n_fft: int,
        d_gate: int = 256,
        use_toeplitz: bool = False,
        dropout_p: float = 0.0,
        pooling_type: str = "dct",
        num_groups: int = 4,
        num_buckets: Optional[int] = None,
        wavelet_on_rate: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [
                SpectreHead(
                    self.head_dim,
                    fft_size=n_fft,
                    d_gate=d_gate,
                    use_toeplitz=use_toeplitz,
                    dropout_p=dropout_p,
                    pooling_type=pooling_type,
                    num_groups=num_groups,
                    num_buckets=num_buckets,
                )
                for _ in range(num_heads)
            ]
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wavelet_refinement = WaveletRefinement(embed_dim, on_rate=wavelet_on_rate)

    def forward(
        self,
        x: torch.Tensor,
        pos_phase: Optional[torch.Tensor] = None,
        memory_fft: Optional[torch.Tensor] = None,
    ):
        chunks = torch.chunk(x, self.num_heads, dim=-1)
        if memory_fft is not None:
            mem_chunks = torch.chunk(memory_fft, self.num_heads, dim=-1)
        else:
            mem_chunks = [None] * self.num_heads
        mixed_and_pools = [
            h(c, pos_phase, return_q_pool=True, memory_fft=m)
            for h, c, m in zip(self.heads, chunks, mem_chunks)
        ]
        mixed = [m for m, _ in mixed_and_pools]
        q_pools = [q for _, q in mixed_and_pools]
        mixed_concat = torch.cat(mixed, dim=-1)
        q_pool_concat = torch.cat(q_pools, dim=-1)
        mixed_refined = self.wavelet_refinement(mixed_concat, q_pool_concat)
        return self.out_proj(mixed_refined)


# ========== SpectreBlock：Transformer即插即用 ===========
class SpectreBlock(nn.Module):
    """
    Drop-in替换Transformer Attention Block.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        n_fft: int,
        mlp_ratio: int = 4,
        d_gate: int = 256,
        use_toeplitz: bool = False,
        dropout_p: float = 0.0,
        pooling_type: str = "dct",
        num_groups: int = 4,
        num_buckets: Optional[int] = None,
        wavelet_on_rate: float = 0.1,
        memory_size: int = 0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mix = SpectreMultiHead(
            embed_dim,
            num_heads,
            n_fft,
            d_gate=d_gate,
            use_toeplitz=use_toeplitz,
            dropout_p=dropout_p,
            pooling_type=pooling_type,
            num_groups=num_groups,
            num_buckets=num_buckets,
            wavelet_on_rate=wavelet_on_rate,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
        )
        if memory_size > 0:
            mem_freq_bins = (
                min(memory_size, n_fft // 2 + 1) if memory_size > 1 else n_fft // 2 + 1
            )
            self.register_parameter(
                "memory_fft",
                nn.Parameter(
                    torch.randn(mem_freq_bins, embed_dim, dtype=torch.cfloat)
                    / math.sqrt(embed_dim)
                ),
            )
            self.memory_fft.requires_grad_(False)
            self.memory_freq_bins = mem_freq_bins
            self.full_freq_bins = n_fft // 2 + 1
        else:
            self.memory_fft = None

    def forward(self, x: torch.Tensor):
        memory_fft = self.memory_fft
        if memory_fft is not None and self.memory_freq_bins < self.full_freq_bins:
            pad_size = self.full_freq_bins - self.memory_freq_bins
            memory_fft = F.pad(
                memory_fft, (0, 0, 0, pad_size), mode="constant", value=0
            )
        x = x + self.mix(self.ln1(x), memory_fft=memory_fft)
        x = x + self.mlp(self.ln2(x))
        return x


# ========== NLP即插即用SPECTRE核心 ===========
class SpectreBlockNLP(nn.Module):
    """
    SPECTRE for NLP: 即插即用频域混合 Attention，可直接替换Transformer自注意力层。
    支持标准Transformer输入 (B, N, d)，attention_mask、position_embeddings，兼容主流NLP库。
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        n_fft=128,
        mlp_ratio=4,
        d_gate=256,
        use_toeplitz=False,
        dropout_p=0.1,
        pooling_type="dct",
        num_groups=4,
        num_buckets=None,
        wavelet_on_rate=0.1,
        memory_size=0,
    ):
        super().__init__()
        self.block = SpectreBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            n_fft=n_fft,
            mlp_ratio=mlp_ratio,
            d_gate=d_gate,
            use_toeplitz=use_toeplitz,
            dropout_p=dropout_p,
            pooling_type=pooling_type,
            num_groups=num_groups,
            num_buckets=num_buckets,
            wavelet_on_rate=wavelet_on_rate,
            memory_size=memory_size,
        )

    def forward(self, x, attention_mask=None, position_embeddings=None):
        """
        Args:
            x: tensor (B, N, d), 输入序列embedding
            attention_mask: tensor (B, N), 可选，padding位置为0
            position_embeddings: tensor (B, N, d), 可选，标准位置编码或频域相位
        Returns:
            tensor (B, N, d), 与自注意力层输出一致
        """
        if position_embeddings is not None:
            x = x + position_embeddings
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).type_as(x)
        return self.block(x)
