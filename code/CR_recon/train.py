"""
CLI 진입점: python train.py --config configs/default.yaml
자동 데이터 정제 기능 포함 (정제된 데이터 없으면 자동 생성)
"""
import argparse
import sys
import subprocess
from pathlib import Path

from utils import load_config
from trainer import Trainer


def ensure_preprocessed_data(cfg_dir):
    """
    정제된 데이터(bayer/*.npy) 존재 여부 확인
    없으면 preprocess_data.py 자동 실행

    Args:
        cfg_dir: CR_recon 디렉토리 경로
    """
    bayer_dir = cfg_dir / "dataset" / "bayer"
    required_files = [
        bayer_dir / "struct_0.npy",
        bayer_dir / "struct_1.npy",
        bayer_dir / "bayer_0.npy",
        bayer_dir / "bayer_1.npy",
        bayer_dir / "bayer_rotated_0.npy",
        bayer_dir / "bayer_rotated_1.npy"
    ]

    # 모든 파일이 존재하는지 확인
    if all(f.exists() for f in required_files):
        print("[INFO] 정제된 데이터가 이미 존재합니다. 재사용합니다.")
        return True

    # 정제된 데이터 없으면 자동 생성
    print("[INFO] 정제된 데이터를 찾을 수 없습니다.")
    print("[INFO] preprocess_data.py를 실행하여 데이터를 정제합니다...")
    print("-" * 80)

    preprocess_script = cfg_dir / "preprocess_data.py"
    if not preprocess_script.exists():
        print(f"[ERROR] preprocess_data.py를 찾을 수 없습니다: {preprocess_script}")
        return False

    try:
        result = subprocess.run(
            ["python", str(preprocess_script)],
            cwd=str(cfg_dir),
            capture_output=False,
            text=True,
            timeout=1800  # 30분 제한시간
        )
        if result.returncode != 0:
            print(f"[ERROR] 데이터 정제 실패 (exit code: {result.returncode})")
            return False

        # 정제 후 파일 존재 확인
        if all(f.exists() for f in required_files):
            print("-" * 80)
            print("[OK] 데이터 정제 완료!")
            return True
        else:
            print("[ERROR] 데이터 정제 후에도 파일이 생성되지 않았습니다.")
            return False

    except subprocess.TimeoutExpired:
        print("[ERROR] 데이터 정제 시간 초과 (30분)")
        return False
    except Exception as e:
        print(f"[ERROR] 데이터 정제 중 오류 발생: {e}")
        return False


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

    # CR_recon 디렉토리
    cfg_file_path = Path(args.config).resolve()
    cfg_dir = cfg_file_path.parent.parent  # configs/default.yaml → CR_recon/

    # 정제된 데이터 확인 및 자동 생성
    if not ensure_preprocessed_data(cfg_dir):
        print("\n[ERROR] 데이터 정제에 실패했습니다. 학습을 시작할 수 없습니다.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("학습 시작")
    print("=" * 80 + "\n")

    # Trainer 생성
    trainer = Trainer(cfg)

    # Dashboard URL 출력
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "localhost"

    port = cfg.get("dashboard", {}).get("port", 8501)
    print("\n" + "=" * 80)
    print("🎯 Dashboard URLs")
    print("=" * 80)
    print(f"📱 Local:     http://localhost:{port}")
    print(f"🌐 Network:   http://{local_ip}:{port}")
    print("=" * 80 + "\n")

    # 학습 실행
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
