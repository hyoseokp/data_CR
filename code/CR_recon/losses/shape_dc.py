"""
Absolute + DC(mean) + Shape(Pearson on derivative) loss.

Motivation:
  - Pearson correlation is shift-invariant (after mean subtraction), so it can match overall
    spectral shape while leaving a global offset error.
  - This loss adds an explicit DC(mean) term to penalize per-channel offsets, and applies
    Pearson correlation to the wavelength derivative to focus it on shape only.

Tensors follow project convention:
  pred, tgt: (B, 2, 2, L)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn


def pearson_corr_loss_1d(pred_1d: torch.Tensor, tgt_1d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Pearson correlation loss along last dim for a flattened (N, L) tensor.
    Returns mean(1 - r).
    """
    p = pred_1d - pred_1d.mean(dim=-1, keepdim=True)
    t = tgt_1d - tgt_1d.mean(dim=-1, keepdim=True)
    r = (p * t).sum(dim=-1) / (p.norm(dim=-1) * t.norm(dim=-1) + eps)
    return (1.0 - r).mean()


def _finite_reduce(x: torch.Tensor, reduce: str) -> torch.Tensor:
    """
    Reduce over all elements, ignoring non-finite values.
    Returns NaN if there are no finite elements.

    Works on older torch builds without torch.nanmin/nanmean/nanmax.
    """
    xf = x[torch.isfinite(x)]
    if xf.numel() == 0:
        return torch.tensor(float("nan"), device=x.device, dtype=torch.float32)
    if reduce == "min":
        return xf.min()
    if reduce == "max":
        return xf.max()
    if reduce == "mean":
        return xf.mean()
    raise ValueError(f"Unknown reduce: {reduce}")


def get_abs_dc_shape_loss(
    w_abs: float = 1.0,
    w_dc: float = 0.5,
    w_shape: float = 0.2,
    beta: float = 0.02,
    blue_weight: float = 1.0,
) -> callable:
    """
    Factory function.

    Args:
      w_abs: weight for absolute value loss (SmoothL1 on values)
      w_dc: weight for per-channel mean-over-wavelength loss (SmoothL1 on DC)
      w_shape: weight for shape loss (Pearson on first derivative over wavelength)
      beta: SmoothL1 beta parameter
      blue_weight: weight multiplier for Blue channel [1,1]
    """
    class AbsDCShapeLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self._last_stats = {}

        def _weighted_smooth_l1(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            if blue_weight == 1.0:
                return F.smooth_l1_loss(pred, tgt, beta=beta)
            w = torch.ones(1, 2, 2, 1, device=pred.device, dtype=pred.dtype)
            w[0, 1, 1, 0] = blue_weight
            loss = F.smooth_l1_loss(pred, tgt, beta=beta, reduction="none")
            return (loss * w).mean()

        def get_last_stats(self) -> dict:
            # Return a copy so callers can't mutate internal state.
            return dict(self._last_stats) if isinstance(self._last_stats, dict) else {}

        def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            pred = pred.float()
            tgt = tgt.float()

            # Numerical stability
            pred = torch.clamp(pred, -1e3, 1e3)
            tgt = torch.clamp(tgt, -1e3, 1e3)

            # Absolute loss (values)
            l_abs = self._weighted_smooth_l1(pred, tgt)

            # DC loss (channel-wise mean across wavelength)
            pred_dc = pred.mean(dim=-1)
            tgt_dc = tgt.mean(dim=-1)
            l_dc = self._weighted_smooth_l1(pred_dc.unsqueeze(-1), tgt_dc.unsqueeze(-1))

            # Shape loss: Pearson correlation on first derivative along wavelength.
            # This can become ill-conditioned when dp/dt norms get very small.
            if pred.shape[-1] >= 2:
                dp = pred[..., 1:] - pred[..., :-1]
                dt = tgt[..., 1:] - tgt[..., :-1]
                B, _, _, Lm1 = dp.shape

                dp_flat = dp.reshape(B * 4, Lm1)
                dt_flat = dt.reshape(B * 4, Lm1)

                eps = 1e-8
                p = dp_flat - dp_flat.mean(dim=-1, keepdim=True)
                t = dt_flat - dt_flat.mean(dim=-1, keepdim=True)
                p_norm = p.norm(dim=-1)
                t_norm = t.norm(dim=-1)
                denom = p_norm * t_norm + eps
                r = (p * t).sum(dim=-1) / denom
                l_shape = (1.0 - r).mean()

                with torch.no_grad():
                    self._last_stats = {
                        "l_abs": float(l_abs.detach().item()),
                        "l_dc": float(l_dc.detach().item()),
                        "l_shape": float(l_shape.detach().item()),
                        "r_mean": float(_finite_reduce(r.detach(), "mean").item()),
                        "r_min": float(_finite_reduce(r.detach(), "min").item()),
                        "dp_norm_mean": float(_finite_reduce(dp_flat.detach().norm(dim=-1), "mean").item()),
                        "dp_norm_min": float(_finite_reduce(dp_flat.detach().norm(dim=-1), "min").item()),
                        "dt_norm_mean": float(_finite_reduce(dt_flat.detach().norm(dim=-1), "mean").item()),
                        "dt_norm_min": float(_finite_reduce(dt_flat.detach().norm(dim=-1), "min").item()),
                        "denom_min": float(_finite_reduce(denom.detach(), "min").item()),
                        "r_finite_frac": float(torch.isfinite(r.detach()).float().mean().item()),
                    }
            else:
                l_shape = pred.new_tensor(0.0)
                with torch.no_grad():
                    self._last_stats = {
                        "l_abs": float(l_abs.detach().item()),
                        "l_dc": float(l_dc.detach().item()),
                        "l_shape": 0.0,
                        "r_mean": float("nan"),
                        "r_min": float("nan"),
                        "dp_norm_mean": float("nan"),
                        "dp_norm_min": float("nan"),
                        "dt_norm_mean": float("nan"),
                        "dt_norm_min": float("nan"),
                        "denom_min": float("nan"),
                        "r_finite_frac": float("nan"),
                    }

            loss = w_abs * l_abs + w_dc * l_dc + w_shape * l_shape
            return torch.clamp(loss, 0, 100)

    return AbsDCShapeLoss()
