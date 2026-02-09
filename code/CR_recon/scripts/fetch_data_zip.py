"""
Download the GitHub repo ZIP and place required .npy data files into the expected folder:
  <repo_root>/data_CR-main/

This is intended as a convenience when the repo was cloned with LFS smudge disabled or
when the data directory is missing.

Note: If Git LFS bandwidth/quota is exceeded, GitHub may serve LFS pointer files instead
of real binaries. This script detects that and will warn/fail accordingly.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path


GITHUB_ZIP_URL = "https://github.com/hyoseokp/data_CR/archive/refs/heads/main.zip"
DATA_FILES = (
    "binary_dataset_128_0.npy",
    "binary_dataset_128_1.npy",
    "spectra_latest_0.npy",
    "spectra_latest_1.npy",
)


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(200)
        return b"version https://git-lfs.github.com/spec/v1" in head
    except OSError:
        return False


def _looks_like_npy(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(6) == b"\x93NUMPY"
    except OSError:
        return False


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb_done = downloaded / (1 << 20)
        mb_total = total_size / (1 << 20)
        print(f"\r[DOWNLOAD] {mb_done:.1f} / {mb_total:.1f} MB ({pct:.0f}%)", end="", flush=True)
    else:
        mb_done = downloaded / (1 << 20)
        print(f"\r[DOWNLOAD] {mb_done:.1f} MB", end="", flush=True)


def download_with_resume(url: str, dest: Path, retries: int = 5, backoff_sec: float = 1.0) -> None:
    """
    Download URL to dest with retry + (best-effort) resume using HTTP Range.
    GitHub supports Range for large files.
    """
    dest = Path(dest)
    for attempt in range(1, retries + 1):
        try:
            existing = dest.stat().st_size if dest.exists() else 0
            headers = {}
            mode = "wb"
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", None)
                # If server ignored Range, restart from scratch.
                if existing > 0 and status != 206:
                    existing = 0
                    mode = "wb"

                total = resp.getheader("Content-Length")
                total = int(total) if total and total.isdigit() else None
                downloaded = existing

                with open(dest, mode) as f:
                    while True:
                        chunk = resp.read(1 << 20)  # 1MB
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total is not None:
                            pct = min(downloaded / (existing + total) * 100, 100)
                            print(
                                f"\r[DOWNLOAD] {downloaded / (1 << 20):.1f} MB ({pct:.0f}%)",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\r[DOWNLOAD] {downloaded / (1 << 20):.1f} MB",
                                end="",
                                flush=True,
                            )
            print()
            return
        except Exception as e:
            if attempt >= retries:
                raise
            wait = backoff_sec * (2 ** (attempt - 1))
            print(f"\n[WARN] Download failed (attempt {attempt}/{retries}): {e}")
            print(f"[INFO] Retrying in {wait:.1f}s...")
            time.sleep(wait)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--url",
        type=str,
        default=GITHUB_ZIP_URL,
        help="GitHub zip URL (default: main branch archive)",
    )
    ap.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Destination data directory (default: <repo_root>/data_CR-main)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination files even if they already exist (still will not overwrite real .npy with LFS pointers)",
    )
    args = ap.parse_args()

    cr_recon_dir = Path(__file__).resolve().parent.parent  # .../code/CR_recon
    repo_root = cr_recon_dir.parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else (repo_root / "data_CR-main")
    data_dir.mkdir(parents=True, exist_ok=True)

    tmp_zip = None
    tmp_dir = None
    try:
        print(f"[INFO] Downloading: {args.url}")
        fd, tmp_zip = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        download_with_resume(args.url, Path(tmp_zip), retries=6, backoff_sec=1.0)

        tmp_dir = Path(tempfile.mkdtemp(prefix="data_cr_zip_"))
        print(f"[INFO] Extracting to: {tmp_dir}")
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(tmp_dir)

        # GitHub archive usually extracts as a single top-level folder.
        subdirs = [p for p in tmp_dir.iterdir() if p.is_dir()]
        if not subdirs:
            print("[ERROR] ZIP extract produced no directories.")
            return 1
        extracted_root = subdirs[0]

        copied = 0
        for name in DATA_FILES:
            src = extracted_root / name
            dst = data_dir / name
            if not src.exists():
                print(f"[ERROR] Missing in ZIP: {name}")
                return 1

            src_is_pointer = _is_lfs_pointer(src)
            src_is_npy = _looks_like_npy(src)

            if not args.force and dst.exists():
                print(f"[INFO] Exists, skip (use --force to overwrite): {dst}")
                continue

            if dst.exists() and _looks_like_npy(dst) and (src_is_pointer or not src_is_npy):
                print(f"[WARN] Refusing to overwrite real data with invalid source: {name}")
                continue

            shutil.copy2(src, dst)
            copied += 1

            if src_is_pointer or not src_is_npy:
                print(f"[WARN] Copied {name}, but it does not look like a real .npy (likely LFS pointer). Size={dst.stat().st_size} bytes")
            else:
                print(f"[OK] Copied {name} -> {dst} ({dst.stat().st_size / (1 << 20):.1f} MB)")

        # Validate destination.
        invalid = []
        for name in DATA_FILES:
            p = data_dir / name
            if not p.exists() or not _looks_like_npy(p) or _is_lfs_pointer(p):
                invalid.append(name)

        if invalid:
            print("\n[ERROR] Data files are still invalid after copy:")
            print("  " + ", ".join(invalid))
            print("\nThis usually means GitHub served Git LFS pointer files instead of real binaries (LFS quota/budget issue).")
            return 2

        print(f"\n[OK] Data ready in: {data_dir} (copied {copied} file(s))")
        return 0

    finally:
        try:
            if tmp_zip and os.path.exists(tmp_zip):
                os.remove(tmp_zip)
        except OSError:
            pass
        try:
            if tmp_dir and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
