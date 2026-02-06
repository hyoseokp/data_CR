"""
Config 로딩 유틸리티
YAML 파일을 읽어 dict로 반환한다.
"""
import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    """YAML config 파일을 로드하여 dict로 반환한다."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
