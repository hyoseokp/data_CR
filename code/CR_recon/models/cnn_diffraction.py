"""
MetaSpec_DiffractionNet: 회절 물리 기반 메타표면 스펙트럼 예측 모델

물리적 배경:
  - 2μm×2μm binary SiN 구조물 (600nm 높이) → 회절/산란 → 2.5μm 하단 RGGB 디텍터
  - 회절 = 2D 푸리에 변환: 구조물의 공간 주파수가 각 파장의 회절각 결정
  - 각 파장 λ는 독립적으로 구조물의 다른 공간 주파수에 반응

아키텍처:
  1. CNN backbone (cnn_xattn과 동일, weight transfer 가능)
  2. FourierFeatureMixing: 16×16에서 2D FFT 기반 전역 공간 믹싱 (회절 모사)
  3. WavelengthEmbedding: 물리적 파장값 기반 sinusoidal 임베딩
  4. DiffractionDecoderBlock: cross-attention(파장→공간) + depthwise 1D conv(파장 축)
     - self-attention 제거 → O(301²) 병목 해소
  5. softplus 출력 (스펙트럼 양수 보장)

입력: (B, 1, 128, 128) float32
출력: (B, 2, 2, out_len) float32
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN backbone 컴포넌트 재사용 (코드 중복 방지)
from .cnn_xattn import _gn, circular_pad_2d, ResBlockGN


# ---------------------------------------------------------------------------
# FourierFeatureMixing: 2D FFT 기반 전역 공간 믹싱
# ---------------------------------------------------------------------------
class FourierFeatureMixing(nn.Module):
    """
    회절 물리를 모사하는 학습 가능한 2D 푸리에 필터.

    rfft2 → 학습 가능한 complex weight로 element-wise 곱 → irfft2
    Residual connection으로 원본 공간 피처 보존.

    16×16 spatial, 128 channels 기준:
      - rfft2 출력: (B, 128, 16, 9) complex
      - 학습 가능 weight: (128, 16, 9, 2) real → 36,864 params
    """

    def __init__(self, channels, spatial_h=16, spatial_w=16):
        super().__init__()
        freq_h = spatial_h
        freq_w = spatial_w // 2 + 1
        # Identity 근처로 초기화 (real ≈ 1, imag ≈ 0)
        self.spectral_weight = nn.Parameter(
            torch.stack([
                torch.ones(channels, freq_h, freq_w),
                torch.zeros(channels, freq_h, freq_w),
            ], dim=-1) + torch.randn(channels, freq_h, freq_w, 2) * 0.02
        )
        self.norm = _gn(channels)

    def forward(self, x):
        """x: (B, C, H, W) → (B, C, H, W) globally mixed via Fourier filtering"""
        identity = x

        # FFT는 float16에서 수치적으로 불안정 → float32 강제
        with torch.amp.autocast("cuda", enabled=False):
            x_f32 = x.float()
            x_fft = torch.fft.rfft2(x_f32, norm="ortho")

            weight = torch.view_as_complex(self.spectral_weight.contiguous())
            x_fft = x_fft * weight.unsqueeze(0)

            x_filtered = torch.fft.irfft2(
                x_fft, s=(x.shape[2], x.shape[3]), norm="ortho"
            )

        x_filtered = x_filtered.to(dtype=identity.dtype)
        return F.silu(self.norm(x_filtered + identity))


# ---------------------------------------------------------------------------
# WavelengthEmbedding: 물리적 파장값 기반 sinusoidal 임베딩
# ---------------------------------------------------------------------------
class WavelengthEmbedding(nn.Module):
    """
    물리적 파장 (400-700 nm)을 d_model 차원 벡터로 인코딩.

    학습된 랜덤 쿼리 대신 sinusoidal 기반:
      - 인접 파장이 자동으로 유사한 임베딩 → 스펙트럼 연속성 inductive bias
      - 학습 가능한 projection으로 물리에 맞게 fine-tuning
    """

    def __init__(self, d_model, out_len=301, lam_min=400.0, lam_max=700.0):
        super().__init__()
        self.d_model = d_model

        # 정규화된 파장 위치: [0.0, ..., 1.0]
        lam = torch.linspace(lam_min, lam_max, out_len)
        lam_norm = (lam - lam_min) / (lam_max - lam_min)
        self.register_buffer("lam_norm", lam_norm)

        # Log-spaced 주파수
        half_d = d_model // 2
        freq_exp = torch.arange(half_d, dtype=torch.float32) / half_d
        freqs = torch.pow(10000.0, -freq_exp)
        self.register_buffer("freqs", freqs)

        # 학습 가능한 projection
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )

    def forward(self):
        """Returns: (out_len, d_model) wavelength embeddings"""
        angles = self.lam_norm.unsqueeze(1) * self.freqs.unsqueeze(0) * 2.0 * math.pi
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.proj(emb)


# ---------------------------------------------------------------------------
# MultiScaleSpatialPool: 16 local + 1 global = 17 공간 토큰
# ---------------------------------------------------------------------------
class MultiScaleSpatialPool(nn.Module):
    """
    CNN 4×4 feature map에서 cross-attention용 공간 토큰 추출.

    16 local tokens: 4×4 grid (RGGB 디텍터 위치 상대적 구조 정보 보존)
    1 global token: average pooling (구조 전체 요약)
    """

    def __init__(self, feat_ch=256, d_model=256):
        super().__init__()
        self.proj = nn.Conv2d(feat_ch, d_model, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_proj = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(feat_ch, d_model),
            nn.SiLU(),
        )
        # 17 positional embeddings (16 local + 1 global)
        self.pos_embed = nn.Parameter(torch.randn(17, 1, d_model) * 0.02)

    def forward(self, f):
        """f: (B, 256, 4, 4) → (17, B, d_model)"""
        B = f.shape[0]
        s = self.proj(f)
        s = s.flatten(2).permute(2, 0, 1).contiguous()  # (16, B, d_model)
        g = self.global_proj(self.pool(f)).unsqueeze(0)  # (1, B, d_model)
        tokens = torch.cat([s, g], dim=0)  # (17, B, d_model)
        return tokens + self.pos_embed


# ---------------------------------------------------------------------------
# DiffractionDecoderBlock: cross-attention + depthwise 1D conv (no self-attention)
# ---------------------------------------------------------------------------
class DiffractionDecoderBlock(nn.Module):
    """
    Decoder block: cross-attention(파장→공간) + depthwise 1D conv(파장 축) + FFN

    self-attention 제거 이유:
      - 파장 간 상관관계는 같은 구조적 특징에서 기인 → cross-attention이 이를 캡처
      - 인접 파장의 로컬 상관관계 → depthwise 1D conv(kernel=15)로 충분
      - O(301²) → O(301×15) 으로 20배 연산 절약

    Fabry-Perot FSR ≈ λ²/(2nT) ≈ 41nm → kernel=15로 반주기 커버,
    4 layer 쌓으면 ERF ≈ 60 bins → 전체 공진 폭 커버.
    """

    def __init__(self, d_model=256, nhead=8, ff_mult=4, dropout=0.1,
                 wav_conv_kernel=15):
        super().__init__()

        # Cross-attention: 파장 queries → 공간 keys/values
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )

        # Depthwise separable 1D conv on wavelength axis
        self.norm_wav = nn.LayerNorm(d_model)
        pad = wav_conv_kernel // 2
        self.wav_dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size=wav_conv_kernel,
            padding=pad, groups=d_model, bias=False
        )
        self.wav_pw_conv = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.wav_act = nn.SiLU()
        self.wav_drop = nn.Dropout(dropout)

        # FFN
        self.norm_ff = nn.LayerNorm(d_model)
        hidden = ff_mult * d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q, memory):
        """
        q: (L, B, D) — 301 wavelength queries
        memory: (17, B, D) — spatial tokens
        """
        # Cross-attention
        x = self.norm_cross(q)
        x2, _ = self.cross_attn(x, memory, memory, need_weights=False)
        q = q + x2

        # Local wavelength mixing: (L, B, D) → (B, D, L) → conv → (L, B, D)
        x = self.norm_wav(q)
        x_t = x.permute(1, 2, 0).contiguous()
        x_t = self.wav_dw_conv(x_t)
        x_t = self.wav_act(x_t)
        x_t = self.wav_pw_conv(x_t)
        x_t = self.wav_drop(x_t)
        q = q + x_t.permute(2, 0, 1)

        # FFN
        x = self.norm_ff(q)
        q = q + self.ff(x)
        return q


# ---------------------------------------------------------------------------
# MetaSpec_DiffractionNet: 전체 모델
# ---------------------------------------------------------------------------
class MetaSpec_DiffractionNet(nn.Module):
    """
    회절 물리 기반 메타표면 스펙트럼 예측 모델.

    CNN backbone → FourierFeatureMixing → MultiScaleSpatialPool
    → WavelengthEmbedding + FiLM → DiffractionDecoderBlock ×N → softplus

    cnn_xattn 대비:
      - self-attention O(301²) 제거 → depthwise 1D conv O(301×15)
      - 2D FFT 공간 믹싱 추가 (회절 물리 모사)
      - sinusoidal 파장 임베딩 (스펙트럼 연속성 inductive bias)
      - 파라미터 ~6.45M (기존 ~7.53M 대비 -14%)
    """

    def __init__(
        self,
        out_len=301,
        d_model=256,
        nhead=8,
        dec_layers=4,
        ff_mult=4,
        cnn_dropout=0.02,
        dec_dropout=0.03,
        head_dropout=0.05,
        wav_conv_kernel=15,
        use_circular_padding=True,
        use_fourier_mixing=True,
        enforce_diag_sym=False,
    ):
        super().__init__()
        self.out_len = out_len
        self.d_model = d_model
        self.use_circular_padding = use_circular_padding
        self.use_fourier_mixing = use_fourier_mixing
        self.enforce_diag_sym = enforce_diag_sym

        # ============================================================
        # ENCODER: CNN backbone (cnn_xattn과 동일 — weight transfer 가능)
        # ============================================================
        self.stem_conv = nn.Conv2d(1, 64, 5, stride=2, padding=0, bias=False)
        self.stem_gn = _gn(64)
        self.stem_dropout = (
            nn.Dropout2d(cnn_dropout) if cnn_dropout > 0 else nn.Identity()
        )

        self.stage1 = nn.Sequential(
            ResBlockGN(64, 96, stride=2, dropout=cnn_dropout,
                       use_circular=use_circular_padding),
            ResBlockGN(96, 96, stride=1, dropout=cnn_dropout,
                       use_circular=use_circular_padding),
        )
        self.stage2 = nn.Sequential(
            ResBlockGN(96, 128, stride=2, dropout=cnn_dropout,
                       use_circular=use_circular_padding),
            ResBlockGN(128, 128, stride=1, dropout=cnn_dropout,
                       use_circular=use_circular_padding),
        )

        # Fourier Feature Mixing at 16×16
        if use_fourier_mixing:
            self.fourier_mix = FourierFeatureMixing(128, spatial_h=16, spatial_w=16)
        else:
            self.fourier_mix = nn.Identity()

        self.stage3 = nn.Sequential(
            ResBlockGN(128, 192, stride=2, dropout=cnn_dropout,
                       use_circular=use_circular_padding),
            ResBlockGN(192, 192, stride=1, dropout=cnn_dropout,
                       use_circular=use_circular_padding),
        )
        self.stage4 = nn.Sequential(
            ResBlockGN(192, 256, stride=2, dropout=cnn_dropout,
                       use_circular=use_circular_padding),
            ResBlockGN(256, 256, stride=1, dropout=cnn_dropout,
                       use_circular=use_circular_padding),
        )

        # ============================================================
        # SPATIAL TOKENS
        # ============================================================
        self.spatial_pool = MultiScaleSpatialPool(feat_ch=256, d_model=d_model)

        # ============================================================
        # WAVELENGTH QUERIES
        # ============================================================
        self.wav_embed = WavelengthEmbedding(d_model, out_len)

        # FiLM: 글로벌 구조 컨텍스트로 파장 쿼리 변조
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.film_proj = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(256, d_model),
            nn.SiLU(),
            nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity(),
        )
        self.film_gate = nn.Linear(d_model, d_model)

        # ============================================================
        # DECODER
        # ============================================================
        self.dec = nn.ModuleList([
            DiffractionDecoderBlock(
                d_model=d_model, nhead=nhead, ff_mult=ff_mult,
                dropout=dec_dropout, wav_conv_kernel=wav_conv_kernel,
            )
            for _ in range(dec_layers)
        ])

        # ============================================================
        # OUTPUT HEAD
        # ============================================================
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity(),
            nn.Linear(d_model, 4),
        )
        self.spec_bias = nn.Parameter(torch.zeros(1, 4, out_len))

    def forward(self, x):
        """
        x: (B, 1, 128, 128) → (B, 2, 2, out_len)
        """
        B = x.shape[0]

        if self.enforce_diag_sym:
            x = 0.5 * (x + x.transpose(-1, -2))

        # === CNN Backbone ===
        if self.use_circular_padding:
            x = circular_pad_2d(x, pad_h=2, pad_w=2)
        else:
            x = F.pad(x, (2, 2, 2, 2), mode="constant", value=0)

        f = F.silu(self.stem_gn(self.stem_conv(x)))
        f = self.stem_dropout(f)              # (B, 64, 64, 64)
        f = self.stage1(f)                    # (B, 96, 32, 32)
        f = self.stage2(f)                    # (B, 128, 16, 16)
        f = self.fourier_mix(f)               # (B, 128, 16, 16) — 2D FFT 믹싱
        f = self.stage3(f)                    # (B, 192, 8, 8)
        f = self.stage4(f)                    # (B, 256, 4, 4)

        # === Spatial Memory Tokens ===
        memory = self.spatial_pool(f)         # (17, B, d_model)

        # === Wavelength Queries ===
        w_emb = self.wav_embed()              # (out_len, d_model)
        q = w_emb.unsqueeze(1).expand(self.out_len, B, -1)  # (L, B, d_model)

        # FiLM modulation
        z = self.film_proj(self.global_pool(f))
        film = self.film_gate(z)
        q = q + film.unsqueeze(0)

        # === Decoder ===
        for blk in self.dec:
            q = blk(q, memory)

        # === Output ===
        y4 = self.head(q).permute(1, 2, 0) + self.spec_bias
        y4 = torch.sigmoid(y4)  # [0, 1] — 물리적 범위 보장
        return y4.view(B, 2, 2, self.out_len)
