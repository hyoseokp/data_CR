"""
Task-01: 데이터 분석 스크립트
로컬 .npy 파일들의 shape/dtype/값 범위/분포를 분석하여 data_summary.md를 생성한다.
"""
import os
import numpy as np
from pathlib import Path

# 데이터 경로 (로컬에 이미 존재하는 경로)
DATA_SRC = Path(r"C:\Users\연구실\gitdata\data_CR")
OUTPUT_DIR = Path(__file__).parent

FILES = {
    "binary_dataset_128_0.npy": "구조 이미지 (파트 0)",
    "binary_dataset_128_1.npy": "구조 이미지 (파트 1)",
    "spectra_latest_0.npy": "스펙트럼 라벨 (파트 0)",
    "spectra_latest_1.npy": "스펙트럼 라벨 (파트 1)",
}


def analyze_file(filepath: Path):
    """numpy 파일을 mmap으로 로드하여 통계를 반환한다."""
    arr = np.load(str(filepath), mmap_mode="r")
    info = {
        "shape": arr.shape,
        "ndim": arr.ndim,
        "dtype": str(arr.dtype),
        "size_mb": os.path.getsize(filepath) / (1024 * 1024),
    }

    # mmap에서 통계 계산 (큰 파일이므로 일부만 사용할 수도 있으나 전체 계산)
    # float로 변환하여 계산
    sample = np.array(arr, dtype=np.float64)
    info["min"] = float(np.min(sample))
    info["max"] = float(np.max(sample))
    info["mean"] = float(np.mean(sample))
    info["std"] = float(np.std(sample))

    # 값 분포 특성
    unique_vals = np.unique(sample[:min(1000, len(sample))].ravel())
    if len(unique_vals) <= 5:
        info["distribution"] = f"Discrete ({len(unique_vals)} unique): {unique_vals.tolist()}"
    else:
        info["distribution"] = f"Continuous ({len(unique_vals)}+ unique values in first 1000 samples)"

    # 유효한 샘플 수 계산 (0이 아닌 값이 있는 샘플)
    if arr.ndim >= 2:
        valid_mask = np.any(arr != 0, axis=tuple(range(1, arr.ndim)))
        valid_count = np.sum(valid_mask)
        info["valid_samples"] = int(valid_count)
        info["zero_samples"] = int(arr.shape[0] - valid_count)
    else:
        info["valid_samples"] = int(arr.shape[0])
        info["zero_samples"] = 0

    return info


def generate_summary(results: dict) -> str:
    """분석 결과를 마크다운 형식으로 생성한다."""
    lines = [
        "# Data Summary",
        "",
        "## 데이터 소스",
        "- GitHub: https://github.com/hyoseokp/data_CR",
        f"- 로컬 경로: `{DATA_SRC}`",
        "",
        "## 파일별 상세",
        "",
        "| 파일명 | 설명 | Shape | Dtype | Size (MB) | Min | Max | Mean | Std | 분포 |",
        "|--------|------|-------|-------|-----------|-----|-----|------|-----|------|",
    ]

    for fname, desc in FILES.items():
        info = results[fname]
        valid_info = f" (유효: {info.get('valid_samples', info['shape'][0])}, 0값: {info.get('zero_samples', 0)})" if 'valid_samples' in info else ""
        lines.append(
            f"| `{fname}` | {desc} | `{info['shape']}` | `{info['dtype']}` | "
            f"{info['size_mb']:.1f} | {info['min']:.4f} | {info['max']:.4f} | "
            f"{info['mean']:.4f} | {info['std']:.4f} | {info['distribution']} |"
        )
        if valid_info:
            lines.append(f"|  | {valid_info} | | | | | | | | |")

    lines.extend([
        "",
        "## 구조 이미지 분석",
        "",
    ])

    # 구조 이미지 분석
    for fname in ["binary_dataset_128_0.npy", "binary_dataset_128_1.npy"]:
        info = results[fname]
        s = info["shape"]
        lines.append(f"### `{fname}`")
        lines.append(f"- Shape: `{s}` → N={s[0]} 샘플")
        if len(s) == 4:
            lines.append(f"- 채널: {s[1]}, 해상도: {s[2]}x{s[3]}")
        elif len(s) == 3:
            lines.append(f"- 해상도: {s[1]}x{s[2]} (채널 없음, 단일 채널 가정)")
        lines.append(f"- 값 범위: [{info['min']}, {info['max']}]")
        lines.append(f"- 분포: {info['distribution']}")
        lines.append("")

    lines.extend([
        "## 스펙트럼 라벨 분석",
        "",
    ])

    # 스펙트럼 분석
    for fname in ["spectra_latest_0.npy", "spectra_latest_1.npy"]:
        info = results[fname]
        s = info["shape"]
        lines.append(f"### `{fname}`")
        lines.append(f"- Shape: `{s}` → N={s[0]} 샘플")
        if len(s) == 3:
            lines.append(f"- 채널: {s[1]} (RGB=3채널 예상), 파장 bin: {s[2]}")
        elif len(s) == 2:
            lines.append(f"- 파장 bin: {s[1]}")
        lines.append(f"- 값 범위: [{info['min']:.6f}, {info['max']:.6f}]")
        lines.append(f"- Mean: {info['mean']:.6f}, Std: {info['std']:.6f}")
        lines.append("")

    # 전체 데이터 규모
    struct_files = ["binary_dataset_128_0.npy", "binary_dataset_128_1.npy"]
    spec_files = ["spectra_latest_0.npy", "spectra_latest_1.npy"]

    total_struct = sum(results[f]["shape"][0] for f in struct_files)
    total_spec = sum(results[f]["shape"][0] for f in spec_files)
    valid_struct = sum(results[f].get("valid_samples", results[f]["shape"][0]) for f in struct_files)
    valid_spec = sum(results[f].get("valid_samples", results[f]["shape"][0]) for f in spec_files)
    total_size = sum(results[f]["size_mb"] for f in FILES)

    lines.extend([
        "## 전체 요약",
        f"- 총 구조 이미지 샘플 수 (원본): {total_struct}",
        f"- 유효한 구조 이미지 샘플 수: {valid_struct}",
        f"- 총 스펙트럼 샘플 수 (원본): {total_spec}",
        f"- 유효한 스펙트럼 샘플 수: {valid_spec}",
        f"- 총 디스크 사용량: {total_size:.1f} MB ({total_size/1024:.2f} GB)",
        f"- 데이터 정제: {total_spec - valid_spec}개 샘플 제외 (모두 0값)",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    print("데이터 분석 시작...")
    results = {}
    for fname in FILES:
        fpath = DATA_SRC / fname
        if not fpath.exists():
            print(f"  [SKIP] {fname} not found at {fpath}")
            continue
        print(f"  분석 중: {fname} ...", end=" ", flush=True)
        results[fname] = analyze_file(fpath)
        print(f"shape={results[fname]['shape']}, dtype={results[fname]['dtype']}")

    if results:
        summary = generate_summary(results)
        out_path = OUTPUT_DIR / "data_summary.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"\ndata_summary.md 생성 완료: {out_path}")
    else:
        print("분석할 파일이 없습니다!")
