"""
Weighted MSE + Smoothness Loss
채널별 가중 MSE + 1차/2차 미분값 loss (spectrum smoothness 정규화)
"""
import torch
import torch.nn.functional as F


def get_weighted_smooth_loss(lambda1=0.1, lambda2=0.05, channel_weights=None):
    """
    Factory function for Weighted MSE + Smoothness loss.

    Args:
        lambda1: weight for 1st derivative (spectrum smoothness)
        lambda2: weight for 2nd derivative (spectrum curvature)
        channel_weights: dict or list for per-channel weights
                        e.g., [1.0, 0.8, 0.8, 1.2] for B, G, G, R

    Returns:
        callable: loss_fn(pred, tgt) -> scalar
    """
    if channel_weights is None:
        channel_weights = [1.0, 0.8, 0.8, 1.2]  # B, G, G, R

    def loss_fn(pred, tgt):
        pred = pred.float()
        tgt = tgt.float()
        device = pred.device

        # Convert channel_weights to tensor
        if isinstance(channel_weights, (list, tuple)):
            w = torch.tensor(channel_weights, dtype=torch.float32, device=device)
        else:
            w = channel_weights

        # Main loss: spectrum values with per-channel weights
        mse = (pred - tgt) ** 2
        weighted_mse = mse * w.view(1, 2, 2, 1)
        value_loss = weighted_mse.mean()

        # 1st derivative loss (spectrum smoothness)
        # Shape: (B, 2, 2, L) → (B, 2, 2, L-1)
        pred_grad1 = torch.diff(pred, dim=-1)
        tgt_grad1 = torch.diff(tgt, dim=-1)
        grad1_mse = (pred_grad1 - tgt_grad1) ** 2
        grad1_loss = grad1_mse.mean()

        # 2nd derivative loss (spectrum curvature)
        # Shape: (B, 2, 2, L-1) → (B, 2, 2, L-2)
        pred_grad2 = torch.diff(pred_grad1, dim=-1)
        tgt_grad2 = torch.diff(tgt_grad1, dim=-1)
        grad2_mse = (pred_grad2 - tgt_grad2) ** 2
        grad2_loss = grad2_mse.mean()

        # Combined loss
        total_loss = value_loss + lambda1 * grad1_loss + lambda2 * grad2_loss
        return total_loss

    return loss_fn
