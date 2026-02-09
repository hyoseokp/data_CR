"""
Create tiny dummy .npy files that match the expected shapes so the training pipeline
can be smoke-tested without downloading large LFS datasets.

Outputs (under <repo_root>/data_CR-main/ by default):
  - binary_dataset_128_0.npy, binary_dataset_128_1.npy  : (N, 1, 128, 128) uint8 {0,1}
  - spectra_latest_0.npy, spectra_latest_1.npy          : (N, 3, 301) float32 (non-zero)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory (default: <repo_root>/data_CR-main)")
    ap.add_argument("--n", type=int, default=32, help="Samples per file")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cr_recon_dir = Path(__file__).resolve().parent.parent  # .../code/CR_recon
    repo_root = cr_recon_dir.parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "data_CR-main")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    def make_struct(n: int) -> np.ndarray:
        # (N,1,128,128) with 0/1 values
        return (rng.random((n, 1, 128, 128)) > 0.5).astype(np.uint8)

    def make_spectra(n: int) -> np.ndarray:
        # (N,3,301) non-zero, smooth-ish curves
        x = np.linspace(0, 1, 301, dtype=np.float32)
        base = np.stack(
            [
                np.sin(2 * np.pi * (x + 0.1)),
                np.sin(2 * np.pi * (x + 0.3)),
                np.sin(2 * np.pi * (x + 0.6)),
            ],
            axis=0,
        )  # (3,301)
        noise = rng.normal(0, 0.05, size=(n, 3, 301)).astype(np.float32)
        amp = rng.uniform(0.5, 1.5, size=(n, 1, 1)).astype(np.float32)
        y = amp * base[None, :, :] + noise
        y = y.astype(np.float32)
        # ensure non-zero by shifting a bit
        y += 0.1
        return y

    for idx in (0, 1):
        struct = make_struct(args.n)
        spectra = make_spectra(args.n)
        np.save(out_dir / f"binary_dataset_128_{idx}.npy", struct)
        np.save(out_dir / f"spectra_latest_{idx}.npy", spectra)

    print(f"[OK] Wrote dummy data to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

