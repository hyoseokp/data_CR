"""
모델 레지스트리
config의 model.name으로 모델을 교체 가능
"""
from .cnn_xattn import MetaSpec_CNNXAttn
from .cnn_gru import MetaSpec_CNNGRU
from .cnn_diffraction import MetaSpec_DiffractionNet
from .se_disco import SE_DiSCO

_MODELS = {
    "cnn_xattn": MetaSpec_CNNXAttn,
    "cnn_gru": MetaSpec_CNNGRU,
    "cnn_diffraction": MetaSpec_DiffractionNet,
    "se_disco": SE_DiSCO,
}


def get_model(name, **kwargs):
    """
    모델 이름으로 인스턴스 생성

    Args:
        name: 모델 이름 (e.g., "cnn_xattn", "cnn_gru", "se_disco")
        **kwargs: 모델 파라미터 (config의 model.params)

    Returns:
        nn.Module 인스턴스
    """
    if name not in _MODELS:
        available = ", ".join(_MODELS.keys())
        raise KeyError(f"Unknown model '{name}'. Available: [{available}]")
    return _MODELS[name](**kwargs)


__all__ = ["get_model", "MetaSpec_CNNXAttn", "MetaSpec_CNNGRU", "MetaSpec_DiffractionNet", "SE_DiSCO"]
