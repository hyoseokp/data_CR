"""
MSE + Pearson Correlation Loss
Per-sample, per-channel Pearson correlation along wavelength dimension.
scale + shift invariant (normalize each channel along wavelength before computing correlation)
"""
import torch
import torch.nn.functional as F


def pearson_corr_loss(pred, tgt, eps=1e-8):
    """
    Per-sample, per-channel Pearson correlation loss along wavelength dim.

    pred, tgt: (B, 2, 2, L) float32
    Returns: scalar loss (1 - r).mean()
    """
    B, _, _, L = pred.shape
    p = pred.reshape(B * 4, L)
    t = tgt.reshape(B * 4, L)

    # Normalize: zero mean
    p = p - p.mean(dim=-1, keepdim=True)
    t = t - t.mean(dim=-1, keepdim=True)

    # Pearson correlation coefficient
    r = (p * t).sum(dim=-1) / (p.norm(dim=-1) * t.norm(dim=-1) + eps)
    return (1.0 - r).mean()


def get_mse_pearson_loss(w_mse=1.0, w_corr=0.2):
    """
    Factory function for MSE + Pearson Correlation loss.

    Args:
        w_mse: weight for MSE loss
        w_corr: weight for Pearson correlation loss

    Returns:
        callable: loss_fn(pred, tgt) -> scalar
    """
    def loss_fn(pred, tgt):
        pred = pred.float()
        tgt = tgt.float()
        l_mse = F.mse_loss(pred, tgt)
        l_corr = pearson_corr_loss(pred, tgt)
        return w_mse * l_mse + w_corr * l_corr

    return loss_fn
