"""
SE-DiSCO: Symmetry-Enhanced Diffraction-aware Spectral Continuous Operator

나노포토닉 컬러 라우팅을 위한 서로게이트 모델.

아키텍처:
  1. Dual-domain 입력: real-space CNN + k-space (log1p|FFT|) CNN → 32×32에서 융합
  2. FNO2D blocks: 학습 가능한 스펙트럴 컨볼루션으로 회절 전역 믹싱
  3. FiLM-SIREN decoder: 연속 파장 implicit 함수로 스펙트럼 생성

물리적 근거:
  - Dual-domain: 실공간 구조 + 주파수 도메인 회절 정보 동시 활용
  - FNO: 회절 = 공간 주파수 필터링 → 주파수 도메인에서 직접 학습
  - SIREN: 스펙트럼의 sharp peak/oscillation을 sin 활성화로 정확히 표현
  - FiLM: 구조 정보로 파장별 스펙트럼 변조 (구조→스펙트럼 매핑)

입력: (B, 1, 128, 128) float32  — binary metasurface
출력: (B, 2, 2, out_len) float32 — BGGR spectra (sigmoid [0,1])
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 유틸리티
# ============================================================================

def _gn(c: int, groups: int = 16) -> nn.GroupNorm:
    """GroupNorm with adaptive group count."""
    g = min(groups, c)
    while c % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, c)


# ============================================================================
# 1. Dual-Domain Encoder
# ============================================================================

class _EncoderBranch(nn.Module):
    """
    경량 ResNet-ish CNN: 128→64→32 (3 conv + 1 res block).
    Real / K-space 양쪽에서 동일 토폴로지 사용.
    """

    def __init__(self, in_ch: int = 1, width: int = 64, dropout: float = 0.02):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            # 128→128 (same)
            nn.Conv2d(in_ch, w // 2, 3, stride=1, padding=1, bias=False),
            _gn(w // 2),
            nn.GELU(),
            # 128→64
            nn.Conv2d(w // 2, w, 3, stride=2, padding=1, bias=False),
            _gn(w),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            # 64→32
            nn.Conv2d(w, w, 3, stride=2, padding=1, bias=False),
            _gn(w),
            nn.GELU(),
        )
        # Residual block at 32×32
        self.res = _ResBlock32(w, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_ch, 128, 128) → (B, width, 32, 32)"""
        return self.res(self.net(x))


class _ResBlock32(nn.Module):
    """32×32 해상도 잔차 블록."""

    def __init__(self, ch: int, dropout: float = 0.02):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.gn1 = _gn(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.gn2 = _gn(ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.gn1(self.conv1(x)))
        h = self.drop(h)
        h = self.gn2(self.conv2(h))
        return F.gelu(h + x)


class DualDomainEncoder(nn.Module):
    """
    Real-space + K-space 이중 인코더.
    - Real: 입력 직접 사용
    - K-space: log1p(|FFT2(x)|) 계산 후 인코딩
    - 32×32에서 concat → 1×1 conv로 융합
    """

    def __init__(self, width: int = 64, dropout: float = 0.02):
        super().__init__()
        self.real_enc = _EncoderBranch(in_ch=1, width=width, dropout=dropout)
        self.kspace_enc = _EncoderBranch(in_ch=1, width=width, dropout=dropout)
        # Fuse: concat(2*width) → width via 1×1 conv
        self.fuse = nn.Sequential(
            nn.Conv2d(width * 2, width, 1, bias=False),
            _gn(width),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 128, 128) → (B, width, 32, 32)"""
        # Real branch
        f_real = self.real_enc(x)

        # K-space branch: FFT → log-magnitude
        with torch.amp.autocast("cuda", enabled=False):
            x_f32 = x.float()
            x_fft = torch.fft.fft2(x_f32, norm="ortho")
            x_mag = torch.log1p(x_fft.abs())  # (B, 1, 128, 128)
        x_mag = x_mag.to(dtype=x.dtype)

        f_kspace = self.kspace_enc(x_mag)

        # Fuse at 32×32
        return self.fuse(torch.cat([f_real, f_kspace], dim=1))


# ============================================================================
# 2. FNO2D Blocks — Fourier Neural Operator for diffraction mixing
# ============================================================================

class SpectralConv2d(nn.Module):
    """
    2D 스펙트럴 컨볼루션 (FNO의 핵심).
    저주파 모드(modes1×modes2)에 대해 학습 가능한 complex weight 적용.

    Factorized: in→rank→out 으로 분해하여 파라미터 절감.
    Full: (in, out, m1, m2) = W²·m² → Factorized: (in, rank, m1, m2) + (rank, out, m1, m2) = 2·W·r·m²
    rank=16, W=64, m=16 기준: 4.2M → 0.52M per block (8× 절감)
    """

    def __init__(self, in_ch: int, out_ch: int, modes1: int = 16, modes2: int = 16,
                 rank: int = 16):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.modes2 = modes2
        self.rank = rank

        scale = 1.0 / (in_ch * rank)
        # Factorized: in→rank  (2개: upper/lower frequency bands)
        self.w1_in = nn.Parameter(torch.randn(in_ch, rank, modes1, modes2, 2) * math.sqrt(scale))
        self.w1_out = nn.Parameter(torch.randn(rank, out_ch, modes1, modes2, 2) * math.sqrt(scale))
        self.w2_in = nn.Parameter(torch.randn(in_ch, rank, modes1, modes2, 2) * math.sqrt(scale))
        self.w2_out = nn.Parameter(torch.randn(rank, out_ch, modes1, modes2, 2) * math.sqrt(scale))

    def _compl_mul2d(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Complex multiplication: (B, in, H, W) × (in, out, H, W) → (B, out, H, W)"""
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        with torch.amp.autocast("cuda", enabled=False):
            x_f32 = x.float()
            x_ft = torch.fft.rfft2(x_f32, norm="ortho")

            w1_in = torch.view_as_complex(self.w1_in.contiguous())
            w1_out = torch.view_as_complex(self.w1_out.contiguous())
            w2_in = torch.view_as_complex(self.w2_in.contiguous())
            w2_out = torch.view_as_complex(self.w2_out.contiguous())

            out_ft = torch.zeros(
                B, self.out_ch, H, W // 2 + 1,
                dtype=torch.cfloat, device=x.device
            )

            m1 = min(self.modes1, H)
            m2 = min(self.modes2, W // 2 + 1)

            # Upper modes (positive freq): in→rank→out
            tmp = self._compl_mul2d(x_ft[:, :, :m1, :m2], w1_in[:, :, :m1, :m2])
            out_ft[:, :, :m1, :m2] = self._compl_mul2d(tmp, w1_out[:, :, :m1, :m2])

            # Lower modes (negative freq, wrapped): in→rank→out
            tmp = self._compl_mul2d(x_ft[:, :, -m1:, :m2], w2_in[:, :, :m1, :m2])
            out_ft[:, :, -m1:, :m2] = self._compl_mul2d(tmp, w2_out[:, :, :m1, :m2])

            out = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")

        return out.to(dtype=x.dtype)


class FNOBlock2d(nn.Module):
    """
    FNO block: spectral conv + 1×1 conv (bypass) + norm + GELU + residual.
    """

    def __init__(self, width: int, modes1: int = 16, modes2: int = 16):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.bypass = nn.Conv2d(width, width, 1, bias=False)
        self.norm = _gn(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.spectral_conv(x) + self.bypass(x)
        h = F.gelu(self.norm(h))
        return h + x  # Residual


# ============================================================================
# 3. FiLM-SIREN Decoder — 연속 파장 implicit spectral decoder
# ============================================================================

class SIRENLayer(nn.Module):
    """SIREN layer with proper initialization."""

    def __init__(self, in_dim: int, out_dim: int, omega0: float = 30.0,
                 is_first: bool = False):
        super().__init__()
        self.omega0 = omega0
        self.linear = nn.Linear(in_dim, out_dim)

        # SIREN initialization (Sitzmann et al., 2020)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)
            else:
                bound = math.sqrt(6.0 / in_dim) / omega0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * self.linear(x))


class FiLMSIREN(nn.Module):
    """
    FiLM-conditioned SIREN MLP.
    구조 latent z로 각 SIREN 레이어를 변조(FiLM)하여
    파장 λ → 스펙트럼 값 매핑.

    FiLM: h = sin(ω₀ · (W·h · (1 + γ) + β))
      γ, β = Linear(z) per layer
    """

    def __init__(
        self,
        z_dim: int = 256,
        hidden_dim: int = 256,
        out_dim: int = 4,        # BGGR 4채널
        n_layers: int = 6,
        omega0: float = 30.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.omega0 = omega0

        # First SIREN layer: λ(1D) → hidden
        self.first_layer = nn.Linear(1, hidden_dim)
        with torch.no_grad():
            self.first_layer.weight.uniform_(-1.0, 1.0)

        # Hidden SIREN layers (FiLM-conditioned)
        self.hidden_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()

        for i in range(n_layers):
            lin = nn.Linear(hidden_dim, hidden_dim)
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_dim) / omega0
                lin.weight.uniform_(-bound, bound)
            self.hidden_layers.append(lin)
            # FiLM generator: z → (gamma, beta)
            self.film_layers.append(nn.Linear(z_dim, hidden_dim * 2))

        # Output layer: no sin activation
        self.out_layer = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega0
            self.out_layer.weight.uniform_(-bound, bound)

        # Initialize FiLM layers near identity: gamma≈0, beta≈0
        for film in self.film_layers:
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

    def forward(self, z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """
        z:   (B, z_dim)       — 구조 latent
        lam: (N_lambda, 1)    — 정규화된 파장 좌표 [-1, 1]

        Returns: (B, N_lambda, out_dim)
        """
        B = z.shape[0]
        N = lam.shape[0]

        # First layer: λ → hidden
        # lam: (N, 1) → expand to (B, N, 1) then broadcast
        h = torch.sin(self.omega0 * self.first_layer(lam))  # (N, hidden)
        h = h.unsqueeze(0).expand(B, -1, -1)  # (B, N, hidden)

        # Hidden layers with FiLM
        for lin, film in zip(self.hidden_layers, self.film_layers):
            film_params = film(z)  # (B, 2*hidden)
            gamma, beta = film_params.chunk(2, dim=-1)  # each (B, hidden)
            gamma = gamma.unsqueeze(1)  # (B, 1, hidden)
            beta = beta.unsqueeze(1)    # (B, 1, hidden)

            h_lin = lin(h)  # (B, N, hidden)
            h = torch.sin(self.omega0 * (h_lin * (1.0 + gamma) + beta))

        # Output: no sin, just linear
        return self.out_layer(h)  # (B, N, out_dim)


# ============================================================================
# 4. SE-DiSCO 전체 모델
# ============================================================================

class SE_DiSCO(nn.Module):
    """
    SE-DiSCO: Symmetry-Enhanced Diffraction-aware Spectral Continuous Operator.

    Dual-domain encoder → FNO2D global mixing → FiLM-SIREN spectral decoder.

    입력: (B, 1, 128, 128) float32
    출력: (B, 2, 2, out_len) float32
    """

    def __init__(
        self,
        out_len: int = 301,
        # Encoder
        width: int = 64,
        enc_dropout: float = 0.02,
        # FNO
        fno_blocks: int = 4,
        fno_modes: int = 16,
        # Latent
        z_dim: int = 256,
        # FiLM-SIREN
        siren_hidden: int = 256,
        siren_layers: int = 6,
        omega0: float = 30.0,
        head_dropout: float = 0.05,
        # Symmetry
        use_tta_symmetry: bool = False,
    ):
        super().__init__()
        self.out_len = out_len
        self.z_dim = z_dim
        self.use_tta_symmetry = use_tta_symmetry

        # === Dual-domain encoder ===
        self.encoder = DualDomainEncoder(width=width, dropout=enc_dropout)

        # === FNO2D blocks at 32×32 ===
        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(width, modes1=fno_modes, modes2=fno_modes)
            for _ in range(fno_blocks)
        ])

        # === Global pool → latent z ===
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to_z = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(width, z_dim),
            nn.GELU(),
            nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity(),
        )

        # === FiLM-SIREN decoder ===
        self.decoder = FiLMSIREN(
            z_dim=z_dim,
            hidden_dim=siren_hidden,
            out_dim=4,  # BGGR 4채널
            n_layers=siren_layers,
            omega0=omega0,
        )

        # === Wavelength grid (buffer, not parameter) ===
        lam_nm = torch.linspace(400.0, 700.0, out_len)
        lam_norm = 2.0 * (lam_nm - 400.0) / (700.0 - 400.0) - 1.0  # [-1, 1]
        self.register_buffer("lam_grid", lam_norm.unsqueeze(-1))  # (out_len, 1)

        # === Learnable output bias (zero init) ===
        self.spec_bias = nn.Parameter(torch.zeros(1, 4, out_len))

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """단일 방향 forward pass. Returns: (B, 2, 2, out_len)"""
        B = x.shape[0]

        # 1. Dual-domain encode → (B, W, 32, 32)
        feat = self.encoder(x)

        # 2. FNO blocks for diffraction mixing
        for blk in self.fno_blocks:
            feat = blk(feat)

        # 3. Global pool → latent z
        z = self.to_z(self.pool(feat))  # (B, z_dim)

        # 4. FiLM-SIREN decode: z + λ → spectra
        # lam_grid: (out_len, 1)
        spectra = self.decoder(z, self.lam_grid)  # (B, out_len, 4)

        # 5. Reshape + bias + sigmoid → [0, 1]
        spectra = spectra.permute(0, 2, 1)  # (B, 4, out_len)
        spectra = spectra + self.spec_bias
        spectra = torch.sigmoid(spectra)

        return spectra.view(B, 2, 2, self.out_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 128, 128) → (B, 2, 2, out_len)
        """
        y = self._forward_single(x)

        if self.use_tta_symmetry and not self.training:
            # TTA: Rot180(x) → predict → swap R↔B → average
            x_rot = torch.rot90(x, 2, [-2, -1])  # 180도 회전
            y_rot = self._forward_single(x_rot)
            # BGGR layout: [[R,G],[G,B]] = (2,2)
            # Swap R(0,0) ↔ B(1,1), G(0,1) ↔ G(1,0)
            y_rot_swapped = y_rot.clone()
            y_rot_swapped[:, 0, 0] = y_rot[:, 1, 1]  # R ← B
            y_rot_swapped[:, 1, 1] = y_rot[:, 0, 0]  # B ← R
            y_rot_swapped[:, 0, 1] = y_rot[:, 1, 0]  # G1 ← G2
            y_rot_swapped[:, 1, 0] = y_rot[:, 0, 1]  # G2 ← G1
            y = 0.5 * (y + y_rot_swapped)

        return y


# ============================================================================
# Factory
# ============================================================================

def get_se_disco(**kwargs) -> SE_DiSCO:
    """Factory function for registry compatibility."""
    return SE_DiSCO(**kwargs)
