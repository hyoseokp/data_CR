"""
CLI 진입점: python train.py --config configs/default.yaml
"""
import argparse
import sys
from pathlib import Path

from utils import load_config
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train MetaSpec model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (YAML)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from"
    )

    args = parser.parse_args()

    # Config 로드
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Trainer 생성
    trainer = Trainer(cfg)

    # 학습 실행
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
