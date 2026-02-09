"""
MSE + Pearson Correlation + L1 loss.

Motivation:
  - Corr matches shape but is shift/scale-invariant and can be numerically unstable on flat signals.
  - MSE/L1 anchor absolute scale/offset.

Tensors follow project convention:
  pred, tgt: (B, 2, 2, L)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pearson_corr_loss(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8, norm_floor: float = 1e-6) -> torch.Tensor:
    """
    Pearson correlation loss along wavelength dim for (B,2,2,L).
    Returns mean(1 - r) over valid channels. If no valid channels, returns 0.
    """
    B, _, _, L = pred.shape
    p = pred.reshape(B * 4, L)
    t = tgt.reshape(B * 4, L)

    p = p - p.mean(dim=-1, keepdim=True)
    t = t - t.mean(dim=-1, keepdim=True)

    p_norm = p.norm(dim=-1)
    t_norm = t.norm(dim=-1)
    valid = (p_norm > norm_floor) & (t_norm > norm_floor)
    denom = p_norm * t_norm + eps
    r = (p * t).sum(dim=-1) / denom
    r = r.clamp(-1.0, 1.0)

    if valid.any():
        return (1.0 - r[valid]).mean()
    return pred.new_tensor(0.0)


class MSECorrL1Loss(nn.Module):
    def __init__(
        self,
        w_mse: float = 1.0,
        w_corr: float = 0.2,
        w_l1: float = 0.1,
        blue_weight: float = 1.0,
        corr_eps: float = 1e-8,
        corr_norm_floor: float = 1e-6,
    ):
        super().__init__()
        self.w_mse = float(w_mse)
        self.w_corr = float(w_corr)
        self.w_l1 = float(w_l1)
        self.blue_weight = float(blue_weight)
        self.corr_eps = float(corr_eps)
        self.corr_norm_floor = float(corr_norm_floor)
        self._last_stats: dict = {}

    def get_last_stats(self) -> dict:
        return dict(self._last_stats) if isinstance(self._last_stats, dict) else {}

    def _blue_weight_tensor(self, pred: torch.Tensor) -> torch.Tensor:
        w = torch.ones(1, 2, 2, 1, device=pred.device, dtype=pred.dtype)
        w[0, 1, 1, 0] = self.blue_weight
        return w

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        tgt = tgt.float()

        # Basic numerical safety. If tensors already contain NaN/Inf, propagate but log it.
        pred = torch.clamp(pred, -1e3, 1e3)
        tgt = torch.clamp(tgt, -1e3, 1e3)

        if self.blue_weight != 1.0:
            w = self._blue_weight_tensor(pred)
            diff = pred - tgt
            l_mse = (w * diff * diff).mean()
            l_l1 = (w * diff.abs()).mean()
        else:
            l_mse = F.mse_loss(pred, tgt)
            l_l1 = F.l1_loss(pred, tgt)

        l_corr = _pearson_corr_loss(pred, tgt, eps=self.corr_eps, norm_floor=self.corr_norm_floor)

        loss = self.w_mse * l_mse + self.w_corr * l_corr + self.w_l1 * l_l1
        loss = torch.clamp(loss, 0, 100)

        with torch.no_grad():
            self._last_stats = {
                "shape_kind": "corr",
                "l_mse": float(l_mse.detach().item()),
                "l_corr": float(l_corr.detach().item()),
                "l_l1": float(l_l1.detach().item()),
            }

        return loss


def get_mse_corr_l1_loss(
    w_mse: float = 1.0,
    w_corr: float = 0.2,
    w_l1: float = 0.1,
    blue_weight: float = 1.0,
    corr_eps: float = 1e-8,
    corr_norm_floor: float = 1e-6,
) -> callable:
    """
    Factory function.
    Returns a nn.Module loss so trainer diagnostics can query get_last_stats().
    """
    return MSECorrL1Loss(
        w_mse=w_mse,
        w_corr=w_corr,
        w_l1=w_l1,
        blue_weight=blue_weight,
        corr_eps=corr_eps,
        corr_norm_floor=corr_norm_floor,
    )

