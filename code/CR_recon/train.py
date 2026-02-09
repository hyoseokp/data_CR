"""
CLI 진입점: python train.py --config configs/default.yaml
자동 데이터 업데이트 및 정제 기능 포함
- GitHub(hyoseokp/data_CR)에서 최신 데이터 자동 다운로드 (zip 방식)
- 정제된 데이터 없으면 자동 생성
"""
import argparse
import hashlib
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
DATA_FILES = (
    "binary_dataset_128_0.npy",
    "binary_dataset_128_1.npy",
    "spectra_latest_0.npy",
    "spectra_latest_1.npy",
)


def _is_lfs_pointer(path: Path) -> bool:
    """
    Detect Git LFS pointer files (small text files that start with the LFS spec header).
    If a repo has exceeded its LFS budget, downloads and zip archives often contain pointers,
    not the actual binary data. Avoid overwriting real local data with pointers.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(200)
        return b"version https://git-lfs.github.com/spec/v1" in head
    except OSError:
        return False


def _looks_like_npy(path: Path) -> bool:
    """Return True if file starts with NumPy .npy magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(6) == b"\x93NUMPY"
    except OSError:
        return False


def validate_data_files(data_dir: Path) -> bool:
    """
    Validate the presence of required data files.
    This catches the common case where the repo uses Git LFS but the actual binaries
    are unavailable (e.g., LFS quota exceeded), leaving small pointer files instead.
    """
    data_dir = Path(data_dir)
    missing = []
    bad = []
    for name in DATA_FILES:
        p = data_dir / name
        if not p.exists():
            missing.append(name)
            continue
        if _is_lfs_pointer(p) or not _looks_like_npy(p):
            bad.append(name)

    if missing or bad:
        print("\n[ERROR] Required dataset files are missing or invalid.")
        if missing:
            print(f"  Missing: {', '.join(missing)}")
        if bad:
            print(f"  Invalid (likely Git LFS pointer, not a real .npy): {', '.join(bad)}")
        print("\nHow to proceed:")
        print("  - If you have the real .npy files, place them under: <repo_root>\\data_CR-main\\")
        print("  - For a quick smoke test without large data:")
        print("      .venv\\Scripts\\python scripts\\make_dummy_data.py --n 16")
        print("      .venv\\Scripts\\python preprocess_data.py")
        print("      .venv\\Scripts\\python train.py --config configs\\smoke.yaml --skip-data-update")
        return False

    return True


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


def ensure_latest_data(data_dir):
    """
    GitHub(hyoseokp/data_CR)에서 최신 데이터를 zip으로 다운로드.
    - .last_update 타임스탬프를 확인하여 1일 미만이면 스킵
    - spectra_latest 파일 변경 시 dataset/bayer/ 삭제 → 재전처리 유도
    - binary_dataset 파일은 로컬에 없을 때만 복사
    - 다운로드 실패 시 경고만 출력하고 기존 데이터로 진행
    """
    data_dir = Path(data_dir)
    last_update_file = data_dir / ".last_update"

    # 1) 최신 여부 확인
    if last_update_file.exists():
        try:
            last_ts = float(last_update_file.read_text().strip())
            elapsed = time.time() - last_ts
            if elapsed < UPDATE_INTERVAL_SEC:
                hours = elapsed / 3600
                print(f"[INFO] 데이터 최신 상태 (마지막 업데이트: {hours:.1f}시간 전)")
                return
        except (ValueError, OSError):
            pass  # 파일 손상 → 업데이트 진행

    print("[INFO] GitHub에서 최신 데이터를 확인합니다...")

    # spectra 파일의 기존 해시 기록 (변경 감지용)
    spectra_files = ["spectra_latest_0.npy", "spectra_latest_1.npy"]
    old_hashes = {}
    for name in spectra_files:
        path = data_dir / name
        if path.exists():
            old_hashes[name] = _md5(path)

    tmp_zip = None
    tmp_dir = None
    try:
        # 2) zip 다운로드
        print(f"[INFO] 다운로드 중: {GITHUB_ZIP_URL}")
        tmp_zip = tempfile.mktemp(suffix=".zip")
        urllib.request.urlretrieve(GITHUB_ZIP_URL, tmp_zip, reporthook=_download_progress)
        print()  # 진행률 줄바꿈

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

        # NOTE:
        # - The GitHub zip contains the whole repository. We only want the data files.
        # - If Git LFS bandwidth/quota is exceeded, these files may be LFS pointers, not real data.
        #   Avoid overwriting existing real local data with pointer files.
        for name in DATA_FILES:
            src = extracted_root / name
            dst = data_dir / name
            if not src.exists():
                continue

            src_is_pointer = _is_lfs_pointer(src)
            if src_is_pointer and dst.exists():
                print(f"[WARN] Skip update for {name}: downloaded file looks like a Git LFS pointer.")
                continue

            # Keep large binary datasets unless missing.
            if name.startswith("binary_dataset") and dst.exists():
                continue

            shutil.copy2(str(src), str(dst))
            if src_is_pointer:
                print(f"[WARN] {name} copied as LFS pointer (not real data).")
            else:
                print(f"[INFO] Updated {name} ({src.stat().st_size / (1 << 20):.1f} MB)")

        # 5) .last_update 기록
        last_update_file.write_text(str(time.time()))
        print("[OK] 데이터 업데이트 완료!")

        # 6) spectra 변경 감지 → dataset/bayer/ 삭제
        spectra_changed = False
        for name in spectra_files:
            path = data_dir / name
            if path.exists():
                new_hash = _md5(path)
                old_hash = old_hashes.get(name)
                if old_hash is None or old_hash != new_hash:
                    spectra_changed = True
                    break
            elif name in old_hashes:
                # 이전에 있었는데 새 zip에 없다면 변경으로 간주
                spectra_changed = True
                break

        if spectra_changed:
            bayer_dir = data_dir.parent / "code" / "CR_recon" / "dataset" / "bayer"
            if bayer_dir.exists():
                print("[INFO] spectra 데이터가 변경되었습니다. 전처리 데이터를 삭제합니다.")
                shutil.rmtree(str(bayer_dir))

    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"\n[WARN] 데이터 다운로드 실패: {e}")
        print("[WARN] 기존 데이터로 진행합니다.")
    except zipfile.BadZipFile:
        print("\n[WARN] 다운로드된 zip 파일이 손상되었습니다.")
        print("[WARN] 기존 데이터로 진행합니다.")
    finally:
        # 임시 파일 정리
        if tmp_zip and os.path.exists(tmp_zip):
            os.remove(tmp_zip)
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


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
        # Raw data might have been downloaded/updated after preprocessing.
        # If so, regenerate the preprocessed bayer files.
        try:
            repo_root = cfg_dir.parent.parent
            data_dir = repo_root / "data_CR-main"
            src_files = [
                data_dir / "binary_dataset_128_0.npy",
                data_dir / "binary_dataset_128_1.npy",
                data_dir / "spectra_latest_0.npy",
                data_dir / "spectra_latest_1.npy",
            ]
            if all(p.exists() for p in src_files):
                latest_src = max(p.stat().st_mtime for p in src_files)
                earliest_bayer = min(p.stat().st_mtime for p in required_files)
                if latest_src > earliest_bayer:
                    print("[INFO] Raw data files look newer than preprocessed outputs. Regenerating dataset/bayer/ ...")
                    shutil.rmtree(str(bayer_dir))
                else:
                    print("[INFO] 정제된 데이터가 이미 존재합니다. 재사용합니다.")
                    return True
            else:
                print("[INFO] 정제된 데이터가 이미 존재합니다. 재사용합니다.")
                return True
        except Exception:
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
            [sys.executable, str(preprocess_script)],
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
    parser.add_argument(
        "--skip-data-update",
        action="store_true",
        help="Skip fetching/updating data from GitHub zip (use local data as-is)"
    )

    args = parser.parse_args()

    # Config 로드
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


    # Runtime environment info (helps debug CPU vs CUDA installs)
    try:
        import torch

        print(f"[INFO] Python: {sys.executable}")
        print(f"[INFO] Torch:  {torch.__version__}")
        print(f"[INFO] CUDA available: {torch.cuda.is_available()} (torch.version.cuda={torch.version.cuda})")
        if torch.cuda.is_available():
            try:
                print(f"[INFO] GPU:   {torch.cuda.get_device_name(0)}")
            except Exception:
                pass
    except Exception as e:
        print(f"[WARN] Failed to print torch runtime info: {e}")

    # CR_recon 디렉토리
    cfg_file_path = Path(args.config).resolve()
    cfg_dir = cfg_file_path.parent.parent  # configs/default.yaml → CR_recon/

    # GitHub에서 최신 데이터 다운로드
    # Canonical location: repo_root/data_CR-main (repo_root is .../data_CR)
    repo_root = cfg_dir.parent.parent
    data_dir = repo_root / "data_CR-main"
    if not args.skip_data_update:
        # If real data already exists locally, don't waste time/bandwidth re-downloading the repo zip.
        if validate_data_files(data_dir):
            try:
                (Path(data_dir) / ".last_update").write_text(str(time.time()))
            except OSError:
                pass
            print("[INFO] Local data files already present; skip GitHub update.")
        else:
            ensure_latest_data(data_dir)
    else:
        print("[INFO] Skip data update (using local data only).")

    if not validate_data_files(data_dir):
        sys.exit(1)

    # 정제된 데이터 확인 및 자동 생성
    if not ensure_preprocessed_data(cfg_dir):
        print("\n[ERROR] 데이터 정제에 실패했습니다. 학습을 시작할 수 없습니다.")
        sys.exit(1)

    # Checkpoints
    outputs_dir = Path(cfg_dir) / "outputs"
    best_ckpt = outputs_dir / f"{cfg['model']['name']}_best.pt"
    last_ckpt = outputs_dir / f"{cfg['model']['name']}_last.pt"

    # Safer default: if checkpoints exist and user didn't specify intent, prefer RESUME over SCRATCH.
    resume_from = args.resume
    init_weights = args.init_weights
    if not resume_from and not init_weights:
        have_best = best_ckpt.exists()
        have_last = last_ckpt.exists()

        if have_best or have_last:
            print("\n[INFO] 기존 checkpoint 발견:")
            if have_last:
                print(f"  - last: {last_ckpt}")
            if have_best:
                print(f"  - best: {best_ckpt}")

            # Non-interactive: don't accidentally start from scratch and overwrite checkpoints.
            if (not sys.stdin) or (not sys.stdin.isatty()):
                if have_last:
                    resume_from = str(last_ckpt)
                    print("[INFO] No stdin available; auto-resume from last checkpoint.")
                else:
                    resume_from = str(best_ckpt)
                    print("[INFO] No stdin available; auto-resume from best checkpoint.")
            else:
                print("\n[선택] 무엇을 할까요?")
                print("  1) Resume from last (권장, optimizer/scheduler/epoch 복원)")
                print("  2) Resume from best (optimizer/scheduler/epoch 복원)")
                print("  3) Init weights from best (epoch 0부터 새로, optimizer 초기화)")
                print("  4) Train from scratch (outputs를 archive로 백업 후 새로 시작)")
                choice = input("번호 입력 [1-4] (기본=1): ").strip() or "1"

                if choice == "1" and have_last:
                    resume_from = str(last_ckpt)
                elif choice == "2" and have_best:
                    resume_from = str(best_ckpt)
                elif choice == "3" and have_best:
                    init_weights = str(best_ckpt)
                elif choice == "4":
                    # Archive existing outputs to avoid accidental overwrite.
                    import shutil
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    archive_dir = outputs_dir / "archive" / ts
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    for p in [best_ckpt, last_ckpt, outputs_dir / "train_log.txt"]:
                        if p.exists():
                            shutil.move(str(p), str(archive_dir / p.name))
                    print(f"[INFO] Archived previous outputs to: {archive_dir}")
                else:
                    # Fallback: safest is resume last if present.
                    if have_last:
                        resume_from = str(last_ckpt)
                        print("[INFO] Invalid choice; fallback to resume last.")
                    elif have_best:
                        resume_from = str(best_ckpt)
                        print("[INFO] Invalid choice; fallback to resume best.")

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
    trainer.train(resume_from=resume_from, init_weights=init_weights)


if __name__ == "__main__":
    main()
