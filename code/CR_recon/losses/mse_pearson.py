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


def get_mse_pearson_loss(w_mse=1.0, w_corr=0.2, blue_weight=1.0):
    """
    Factory function for MSE + Pearson Correlation loss.

    Args:
        w_mse: weight for MSE loss
        w_corr: weight for Pearson correlation loss
        blue_weight: weight multiplier for Blue channel [1,1] in Bayer pattern

    Returns:
        callable: loss_fn(pred, tgt) -> scalar
    """
    def loss_fn(pred, tgt):
        pred = pred.float()
        tgt = tgt.float()

        # Numerical stability: clamp extreme values
        pred = torch.clamp(pred, -1e3, 1e3)
        tgt = torch.clamp(tgt, -1e3, 1e3)

        # Channel-weighted MSE: Blue 채널에 추가 가중치
        if blue_weight != 1.0:
            ch_weight = torch.ones(1, 2, 2, 1, device=pred.device)
            ch_weight[0, 1, 1, 0] = blue_weight
            l_mse = (ch_weight * (pred - tgt) ** 2).mean()
        else:
            l_mse = F.mse_loss(pred, tgt)

        l_corr = pearson_corr_loss(pred, tgt)

        # Clamp final loss to prevent explosion
        loss = w_mse * l_mse + w_corr * l_corr
        loss = torch.clamp(loss, 0, 100)

        return loss

    return loss_fn
