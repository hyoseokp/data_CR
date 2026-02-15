"""
Multi-Scale Spectral Loss.

전체적인 경향성(저주파)과 고주파 성분(피크, 급격한 변화)을 동시에 학습하기 위한
multi-scale loss 함수.

5개 term 결합:
  1. l_abs:   SmoothL1 on values         — 절대값 정확도
  2. l_corr:  Pearson correlation         — 전체 형태 (저주파 경향성)
  3. l_grad1: SmoothL1 on 1st derivative  — 기울기 매칭 (고주파)
  4. l_grad2: SmoothL1 on 2nd derivative  — 곡률 매칭 (고주파)
  5. l_fft:   L1 on high-freq FFT mag     — FFT 고주파 직접 제어

Tensors follow project convention:
  pred, tgt: (B, 2, 2, L)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiscaleSpectralLoss(nn.Module):
    def __init__(
        self,
        w_abs: float = 1.0,
        w_corr: float = 0.3,
        w_grad1: float = 0.5,
        w_grad2: float = 0.2,
        w_fft: float = 0.1,
        blue_weight: float = 2.0,
        beta: float = 0.02,
        fft_start_ratio: float = 0.5,
        corr_eps: float = 1e-8,
        corr_norm_floor: float = 1e-6,
    ):
        super().__init__()
        self.w_abs = float(w_abs)
        self.w_corr = float(w_corr)
        self.w_grad1 = float(w_grad1)
        self.w_grad2 = float(w_grad2)
        self.w_fft = float(w_fft)
        self.blue_weight = float(blue_weight)
        self.beta = float(beta)
        self.fft_start_ratio = float(fft_start_ratio)
        self.corr_eps = float(corr_eps)
        self.corr_norm_floor = float(corr_norm_floor)
        self._last_stats: dict = {}

    def get_last_stats(self) -> dict:
        return dict(self._last_stats) if isinstance(self._last_stats, dict) else {}

    def _blue_weight_tensor(self, ref: torch.Tensor) -> torch.Tensor:
        w = torch.ones(1, 2, 2, 1, device=ref.device, dtype=ref.dtype)
        w[0, 1, 1, 0] = self.blue_weight
        return w

    def _weighted_smooth_l1(self, pred: torch.Tensor, tgt: torch.Tensor, beta: float | None = None) -> torch.Tensor:
        b = self.beta if beta is None else float(beta)
        if self.blue_weight == 1.0:
            return F.smooth_l1_loss(pred, tgt, beta=b)
        w = self._blue_weight_tensor(pred)
        loss = F.smooth_l1_loss(pred, tgt, beta=b, reduction="none")
        return (loss * w).mean()

    def _pearson_corr_loss(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        B, _, _, L = pred.shape
        p = pred.reshape(B * 4, L)
        t = tgt.reshape(B * 4, L)

        p = p - p.mean(dim=-1, keepdim=True)
        t = t - t.mean(dim=-1, keepdim=True)

        p_norm = p.norm(dim=-1)
        t_norm = t.norm(dim=-1)
        valid = (p_norm > self.corr_norm_floor) & (t_norm > self.corr_norm_floor)
        denom = p_norm * t_norm + self.corr_eps
        r = (p * t).sum(dim=-1) / denom
        r = r.clamp(-1.0, 1.0)

        if valid.any():
            return (1.0 - r[valid]).mean()
        return pred.new_tensor(0.0)

    def _fft_high_freq_loss(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        B, _, _, L = pred.shape
        p = pred.reshape(B * 4, L)
        t = tgt.reshape(B * 4, L)

        # rfft → magnitude
        p_fft = torch.fft.rfft(p, dim=-1).abs()
        t_fft = torch.fft.rfft(t, dim=-1).abs()

        # 고주파 대역 선택 (상위 fft_start_ratio)
        n_freq = p_fft.shape[-1]
        start = int(n_freq * self.fft_start_ratio)
        if start >= n_freq:
            return pred.new_tensor(0.0)

        p_hf = p_fft[:, start:]
        t_hf = t_fft[:, start:]

        return F.l1_loss(p_hf, t_hf)

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        tgt = tgt.float()
        pred = torch.clamp(pred, -1e3, 1e3)
        tgt = torch.clamp(tgt, -1e3, 1e3)

        # 1. Absolute value loss
        l_abs = self._weighted_smooth_l1(pred, tgt)

        # 2. Pearson correlation loss (전체 형태)
        l_corr = self._pearson_corr_loss(pred, tgt)

        # 3. 1st derivative loss (기울기 — 고주파)
        dp = pred[..., 1:] - pred[..., :-1]
        dt = tgt[..., 1:] - tgt[..., :-1]
        l_grad1 = self._weighted_smooth_l1(dp, dt)

        # 4. 2nd derivative loss (곡률 — 고주파)
        d2p = dp[..., 1:] - dp[..., :-1]
        d2t = dt[..., 1:] - dt[..., :-1]
        l_grad2 = self._weighted_smooth_l1(d2p, d2t)

        # 5. FFT high-frequency loss
        l_fft = self._fft_high_freq_loss(pred, tgt)

        loss = (
            self.w_abs * l_abs
            + self.w_corr * l_corr
            + self.w_grad1 * l_grad1
            + self.w_grad2 * l_grad2
            + self.w_fft * l_fft
        )

        with torch.no_grad():
            self._last_stats = {
                "shape_kind": "multiscale",
                "l_abs": float(l_abs.detach().item()),
                "l_dc": float(l_corr.detach().item()),    # map to l_dc for trainer compat
                "l_shape": float(l_grad1.detach().item()), # map to l_shape for trainer compat
                "l_grad2": float(l_grad2.detach().item()),
                "l_fft": float(l_fft.detach().item()),
            }

        return torch.clamp(loss, 0, 100)


def get_multiscale_spectral_loss(
    w_abs: float = 1.0,
    w_corr: float = 0.3,
    w_grad1: float = 0.5,
    w_grad2: float = 0.2,
    w_fft: float = 0.1,
    blue_weight: float = 2.0,
    beta: float = 0.02,
    fft_start_ratio: float = 0.5,
    corr_eps: float = 1e-8,
    corr_norm_floor: float = 1e-6,
) -> callable:
    """Factory function."""
    return MultiscaleSpectralLoss(
        w_abs=w_abs,
        w_corr=w_corr,
        w_grad1=w_grad1,
        w_grad2=w_grad2,
        w_fft=w_fft,
        blue_weight=blue_weight,
        beta=beta,
        fft_start_ratio=fft_start_ratio,
        corr_eps=corr_eps,
        corr_norm_floor=corr_norm_floor,
    )
