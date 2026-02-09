"""
CLI 진입점: python train.py --config configs/default.yaml
자동 데이터 업데이트 및 정제 기능 포함
- GitHub(hyoseokp/data_CR)에서 최신 데이터 자동 다운로드 (zip 방식)
- 정제된 데이터 없으면 자동 생성
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from utils import load_config
from trainer import Trainer

GITHUB_ZIP_URL = "https://github.com/hyoseokp/data_CR/archive/refs/heads/main.zip"
UPDATE_INTERVAL_SEC = 86400  # 1일


def _md5(filepath):
    """파일의 MD5 해시를 계산한다."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1 << 20)  # 1MB
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download_progress(block_num, block_size, total_size):
    """urllib.request.urlretrieve 진행률 콜백."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb_done = downloaded / (1 << 20)
        mb_total = total_size / (1 << 20)
        print(f"\r[DOWNLOAD] {mb_done:.1f} / {mb_total:.1f} MB ({pct:.0f}%)", end="", flush=True)
    else:
        mb_done = downloaded / (1 << 20)
        print(f"\r[DOWNLOAD] {mb_done:.1f} MB", end="", flush=True)


def ensure_latest_data(data_dir, cfg_dir=None):
    """
    GitHub(hyoseokp/data_CR)에서 최신 데이터를 zip으로 다운로드.
    - .last_update 타임스탬프를 확인하여 1일 미만이면 스킵
    - spectra_latest 파일 변경 시 dataset/bayer/ 삭제 → 재전처리 유도
    - binary_dataset 파일은 로컬에 없을 때만 복사
    - 다운로드 실패 시 경고만 출력하고 기존 데이터로 진행

    Args:
        data_dir: data_CR-main 디렉토리 경로
        cfg_dir: CR_recon 디렉토리 경로 (spectra 변경 감지 시 bayer/ 삭제용)
    """
    data_dir = Path(data_dir)
    last_update_file = data_dir / ".last_update"

    # spectra 해시 저장 파일
    spectra_hash_file = data_dir / ".spectra_hash"

    # 1) 최신 여부 확인
    if last_update_file.exists():
        try:
            last_ts = float(last_update_file.read_text().strip())
            elapsed = time.time() - last_ts
            if elapsed < UPDATE_INTERVAL_SEC:
                hours = elapsed / 3600
                print(f"[INFO] 데이터 최신 상태 (마지막 업데이트: {hours:.1f}시간 전)")
                return False  # spectra 변경 없음
        except (ValueError, OSError):
            pass  # 파일 손상 → 업데이트 진행

    print("[INFO] GitHub에서 최신 데이터를 확인합니다...")

    # spectra 파일의 기존 해시 기록 (변경 감지용)
    spectra_files = ["spectra_latest_0.npy", "spectra_latest_1.npy"]
    old_hashes = {}
    if spectra_hash_file.exists():
        try:
            import json
            old_hashes = json.loads(spectra_hash_file.read_text())
        except:
            pass

    # 현재 파일의 해시
    for name in spectra_files:
        path = data_dir / name
        if path.exists():
            old_hashes[name] = _md5(path)

    tmp_zip = None
    tmp_dir = None
    try:
        # 2) zip 다운로드 (타임아웃 120초, 3회 재시도)
        print(f"[INFO] 다운로드 중: {GITHUB_ZIP_URL}")
        tmp_zip = tempfile.mktemp(suffix=".zip")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # socket 타임아웃 설정 (120초)
                import socket
                socket.setdefaulttimeout(120)

                urllib.request.urlretrieve(GITHUB_ZIP_URL, tmp_zip, reporthook=_download_progress)
                print()  # 진행률 줄바꿈
                break  # 성공하면 루프 탈출
            except (TimeoutError, urllib.error.URLError, OSError) as e:
                if attempt < max_retries - 1:
                    print(f"\n[WARN] 다운로드 실패 (시도 {attempt+1}/{max_retries}): {e}")
                    print("[INFO] 3초 후 재시도합니다...")
                    time.sleep(3)
                else:
                    raise  # 마지막 시도 실패 → 상위 except로

        # 3) 압축 해제
        tmp_dir = tempfile.mkdtemp()
        print("[INFO] 압축 해제 중...")
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(tmp_dir)

        # zip 내부 구조: data_CR-main/ 디렉토리
        extracted_root = Path(tmp_dir) / "data_CR-main"
        if not extracted_root.is_dir():
            # 혹시 이름이 다를 경우 첫 번째 디렉토리 사용
            subdirs = [d for d in Path(tmp_dir).iterdir() if d.is_dir()]
            if subdirs:
                extracted_root = subdirs[0]
            else:
                print("[WARN] zip 내부 구조를 인식할 수 없습니다. 업데이트를 건너뜁니다.")
                return

        # 4) 파일 복사
        data_dir.mkdir(parents=True, exist_ok=True)

        # binary_dataset 파일: 로컬에 없을 때만 복사 (대용량)
        binary_files = ["binary_dataset_128_0.npy", "binary_dataset_128_1.npy"]
        for name in binary_files:
            src = extracted_root / name
            dst = data_dir / name
            if src.exists() and not dst.exists():
                print(f"[INFO] 복사 중: {name} ({src.stat().st_size / (1 << 30):.1f} GB)")
                shutil.copy2(str(src), str(dst))

        # spectra 및 기타 파일: 항상 복사 (덮어쓰기)
        for item in extracted_root.iterdir():
            if item.name in binary_files:
                continue  # 위에서 처리
            dst = data_dir / item.name
            if item.is_file():
                shutil.copy2(str(item), str(dst))
            elif item.is_dir():
                if dst.exists():
                    shutil.rmtree(str(dst))
                shutil.copytree(str(item), str(dst))

        # 5) .last_update 기록
        last_update_file.write_text(str(time.time()))
        print("[OK] 데이터 업데이트 완료!")

        # 6) spectra 변경 감지 → dataset/bayer/ 삭제
        spectra_changed = False
        new_hashes = {}
        for name in spectra_files:
            path = data_dir / name
            if path.exists():
                new_hash = _md5(path)
                new_hashes[name] = new_hash
                old_hash = old_hashes.get(name)
                if old_hash is None or old_hash != new_hash:
                    spectra_changed = True
                    print(f"[INFO] {name}가 변경되었습니다.")
            elif name in old_hashes:
                # 이전에 있었는데 새 zip에 없다면 변경으로 간주
                spectra_changed = True
                print(f"[INFO] {name}이 제거되었습니다.")

        # spectra 해시 저장
        if new_hashes:
            spectra_hash_file.write_text(json.dumps(new_hashes))

        if spectra_changed:
            # cfg_dir이 주어진 경우 dataset/bayer/ 삭제
            if cfg_dir:
                bayer_dir = Path(cfg_dir) / "dataset" / "bayer"
                if bayer_dir.exists():
                    print("[INFO] spectra 데이터가 변경되었습니다. 전처리 데이터를 삭제합니다.")
                    shutil.rmtree(str(bayer_dir))
                    return True  # 재전처리 필요
            else:
                print("[INFO] spectra 데이터가 변경되었습니다.")

    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
        print(f"\n[WARN] 데이터 다운로드 최종 실패: {type(e).__name__}: {e}")
        print("[WARN] 가능한 원인:")
        print("       - 네트워크 연결 끊김")
        print("       - GitHub 서버 응답 느림")
        print("       - 파일 크기가 매우 큼 (수 GB)")
        print("       - 방화벽 차단")
        print("[WARN] 기존 데이터로 학습을 계속합니다.")
    except zipfile.BadZipFile:
        print("\n[WARN] 다운로드된 zip 파일이 손상되었습니다.")
        print("[WARN] 기존 데이터로 진행합니다.")
    finally:
        # 임시 파일 정리
        if tmp_zip and os.path.exists(tmp_zip):
            os.remove(tmp_zip)
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return False  # 다운로드 실패 또는 spectra 변경 없음


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
    parser.add_argument(
        "--init-weights",
        type=str,
        default=None,
        help="Path to checkpoint to load model weights only (train from epoch 0)"
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

    # GitHub에서 최신 데이터 다운로드 (spectra 변경 시 재전처리)
    data_dir = cfg_dir.parent / "data_CR-main"
    spectra_changed = ensure_latest_data(data_dir, cfg_dir) or False

    # 정제된 데이터 확인 및 자동 생성
    # spectra가 변경되었으면 무조건 재전처리
    if spectra_changed or not ensure_preprocessed_data(cfg_dir):
        print("\n[ERROR] 데이터 정제에 실패했습니다. 학습을 시작할 수 없습니다.")
        sys.exit(1)

    # Best checkpoint 확인 및 사용자 선택
    init_weights = args.init_weights
    if not args.resume and not args.init_weights:
        best_ckpt = Path(cfg_dir) / "outputs" / f"{cfg['model']['name']}_best.pt"
        if best_ckpt.exists():
            print(f"\n[INFO] 기존 best checkpoint 발견: {best_ckpt}")
            choice = input("[선택] 기존 best 파라미터를 불러올까요? (y/n): ").strip().lower()
            if choice == 'y':
                init_weights = str(best_ckpt)
                print(f"[INFO] best 파라미터를 불러와서 학습합니다.")
            else:
                print("[INFO] 처음부터 새로 학습합니다.")

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
    print("Dashboard URLs")
    print("=" * 80)
    print(f"  Local:     http://localhost:{port}")
    print(f"  Network:   http://{local_ip}:{port}")
    print("=" * 80 + "\n")

    # 학습 실행
    trainer.train(resume_from=args.resume, init_weights=init_weights)


if __name__ == "__main__":
    main()
