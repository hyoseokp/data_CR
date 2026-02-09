"""
Output constraint utilities.

Project convention:
  - Model outputs a Bayer-like tensor shaped (B, 2, 2, L) where positions are:
      [0,0]=R, [0,1]=G1, [1,0]=G2, [1,1]=B
  - The dataset returns targets in the same sign space used for training (see data/dataset.py).
    If you want to enforce constraints in a different sign convention, use `physical_is_negative`.
"""

from __future__ import annotations

import torch


def enforce_intensity_sum_range(
    pred: torch.Tensor,
    sum_min: float = 0.45,
    sum_max: float = 0.95,
    physical_is_negative: bool = False,
) -> torch.Tensor:
    """
    Enforce for each wavelength bin k:
        R + (G1 + G2)/2 + B  in [sum_min, sum_max]

    This is a hard projection implemented as a per-bin uniform shift across all 4 channels.
    Because the sum uses weights [1, 0.5, 0.5, 1], adding `a` to all four entries changes
    the weighted sum by 3*a, so we can satisfy the constraint exactly by choosing:
        a = (clamp(total) - total) / 3

    Args:
      pred: (B, 2, 2, L)
      sum_min/sum_max: desired bounds (in physical domain)
      physical_is_negative: if True, interpret physical intensities as `-pred`
    """
    if pred.ndim != 4 or pred.shape[1:3] != (2, 2):
        return pred

    phys = -pred if physical_is_negative else pred

    r = phys[:, 0, 0, :]
    g1 = phys[:, 0, 1, :]
    g2 = phys[:, 1, 0, :]
    b = phys[:, 1, 1, :]

    total = r + 0.5 * (g1 + g2) + b
    target = total.clamp(sum_min, sum_max)

    shift = (target - total) / 3.0  # because weights sum to 3
    phys = phys + shift[:, None, None, :]

    return -phys if physical_is_negative else phys
