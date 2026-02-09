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
    corr_eps: float = 1e-8,
    corr_norm_floor: float = 1e-3,
    corr_denom_floor: float = 1e-4,
    shape_kind: str = "smoothl1",
    shape_beta: float = 0.02,
) -> callable:
    """
    Factory function.

    Args:
      w_abs: weight for absolute value loss (SmoothL1 on values)
      w_dc: weight for per-channel mean-over-wavelength loss (SmoothL1 on DC)
      w_shape: weight for shape loss (Pearson on first derivative over wavelength)
      beta: SmoothL1 beta parameter
      blue_weight: weight multiplier for Blue channel [1,1]
      corr_eps: epsilon added to correlation denominator
      corr_norm_floor: if either norm is below this, skip shape term for that sample
      corr_denom_floor: clamp denominator to at least this to avoid ill-conditioned division
      shape_kind: shape term type over wavelength-derivative. One of: "smoothl1", "mse", "pearson".
      shape_beta: SmoothL1 beta for derivative loss when shape_kind="smoothl1".
    """
    class AbsDCShapeLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self._last_stats = {}

        def _weighted_smooth_l1(self, pred: torch.Tensor, tgt: torch.Tensor, beta_override: float | None = None) -> torch.Tensor:
            b = beta if beta_override is None else float(beta_override)
            if blue_weight == 1.0:
                return F.smooth_l1_loss(pred, tgt, beta=b)
            w = torch.ones(1, 2, 2, 1, device=pred.device, dtype=pred.dtype)
            w[0, 1, 1, 0] = blue_weight
            loss = F.smooth_l1_loss(pred, tgt, beta=b, reduction="none")
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
            # Default is derivative SmoothL1 (more stable than Pearson corr).
            if pred.shape[-1] >= 2:
                dp = pred[..., 1:] - pred[..., :-1]
                dt = tgt[..., 1:] - tgt[..., :-1]
                B, _, _, Lm1 = dp.shape

                if shape_kind == "pearson":
                    # Kept for backward-compatibility; may be unstable.
                    dp_flat = dp.reshape(B * 4, Lm1)
                    dt_flat = dt.reshape(B * 4, Lm1)

                    p = dp_flat - dp_flat.mean(dim=-1, keepdim=True)
                    t = dt_flat - dt_flat.mean(dim=-1, keepdim=True)
                    p_norm = p.norm(dim=-1)
                    t_norm = t.norm(dim=-1)
                    denom_raw = p_norm * t_norm
                    valid = (p_norm > corr_norm_floor) & (t_norm > corr_norm_floor)
                    denom = torch.clamp(denom_raw, min=corr_denom_floor) + corr_eps

                    r = (p * t).sum(dim=-1) / denom
                    r = r.clamp(-1.0, 1.0)

                    if valid.any():
                        r_valid = r[valid]
                        l_shape = (1.0 - r_valid).mean()
                        valid_frac = float(valid.float().mean().item())
                    else:
                        l_shape = pred.new_tensor(0.0)
                        valid_frac = 0.0

                    with torch.no_grad():
                        self._last_stats = {
                            "shape_kind": "pearson",
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
                            "shape_valid_frac": float(valid_frac),
                            "r_finite_frac": float(torch.isfinite(r.detach()).float().mean().item()),
                        }
                else:
                    # Stable shape: match first derivative directly.
                    if shape_kind == "mse":
                        w = torch.ones(1, 2, 2, 1, device=dp.device, dtype=dp.dtype)
                        if blue_weight != 1.0:
                            w[0, 1, 1, 0] = blue_weight
                        diff2 = (dp - dt) ** 2
                        l_shape = (diff2 * w).mean()
                        d1_err = torch.sqrt(diff2 + 1e-12)
                    else:
                        # default: smoothl1
                        l_shape = self._weighted_smooth_l1(dp, dt, beta_override=shape_beta)
                        d1_err = (dp - dt).abs()

                    with torch.no_grad():
                        self._last_stats = {
                            "shape_kind": "mse" if shape_kind == "mse" else "smoothl1",
                            "l_abs": float(l_abs.detach().item()),
                            "l_dc": float(l_dc.detach().item()),
                            "l_shape": float(l_shape.detach().item()),
                            "d1_err_mean": float(d1_err.detach().mean().item()),
                            "d1_err_max": float(d1_err.detach().max().item()),
                        }
            else:
                l_shape = pred.new_tensor(0.0)
                with torch.no_grad():
                    self._last_stats = {
                        "shape_kind": shape_kind,
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
                        "shape_valid_frac": float("nan"),
                        "r_finite_frac": float("nan"),
                        "d1_err_mean": float("nan"),
                        "d1_err_max": float("nan"),
                    }

            loss = w_abs * l_abs + w_dc * l_dc + w_shape * l_shape
            return torch.clamp(loss, 0, 100)

    return AbsDCShapeLoss()
