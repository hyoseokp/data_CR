"""
Loss 함수 레지스트리
config의 loss.name으로 loss를 교체 가능
"""
from .mse_pearson import get_mse_pearson_loss
from .mse_corr_l1 import get_mse_corr_l1_loss
from .shape_dc import get_abs_dc_shape_loss
from .weighted_smooth import get_weighted_smooth_loss

_LOSSES = {
    "mse_pearson": get_mse_pearson_loss,
    "mse_corr_l1": get_mse_corr_l1_loss,
    "abs_dc_shape": get_abs_dc_shape_loss,
    "weighted_smooth": get_weighted_smooth_loss,
}


def get_loss(name, **kwargs):
    """
    Loss 이름으로 callable 함수 반환

    Args:
        name: loss 함수 이름 (e.g., "mse_pearson", "weighted_smooth")
        **kwargs: loss 파라미터 (config의 loss.params)

    Returns:
        callable: loss_fn(pred, tgt) -> scalar
    """
    if name not in _LOSSES:
        available = ", ".join(_LOSSES.keys())
        raise KeyError(f"Unknown loss '{name}'. Available: [{available}]")
    return _LOSSES[name](**kwargs)


__all__ = ["get_loss"]
