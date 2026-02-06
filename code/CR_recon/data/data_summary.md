# Data Summary

## 데이터 소스
- GitHub: https://github.com/hyoseokp/data_CR
- 로컬 경로: `C:\Users\연구실\gitdata\data_CR`

## 파일별 상세

| 파일명 | 설명 | Shape | Dtype | Size (MB) | Min | Max | Mean | Std | 분포 |
|--------|------|-------|-------|-----------|-----|-----|------|-----|------|
| `binary_dataset_128_0.npy` | 구조 이미지 (파트 0) | `(100000, 1, 128, 128)` | `uint8` | 1562.5 | 0.0000 | 1.0000 | 0.3229 | 0.4676 | Discrete (2 unique): [0.0, 1.0] |
| `binary_dataset_128_1.npy` | 구조 이미지 (파트 1) | `(100000, 1, 128, 128)` | `uint8` | 1562.5 | 0.0000 | 1.0000 | 0.3211 | 0.4669 | Discrete (2 unique): [0.0, 1.0] |
| `spectra_latest_0.npy` | 스펙트럼 라벨 (파트 0) | `(100000, 3, 301)` | `float32` | 344.5 | -0.6710 | 0.0000 | -0.1193 | 0.1257 | Continuous (821522+ unique values in first 1000 samples) |
| `spectra_latest_1.npy` | 스펙트럼 라벨 (파트 1) | `(100000, 3, 301)` | `float32` | 344.5 | -0.6339 | 0.0000 | -0.1074 | 0.1244 | Continuous (863036+ unique values in first 1000 samples) |

## 구조 이미지 분석

### `binary_dataset_128_0.npy`
- Shape: `(100000, 1, 128, 128)` → N=100000 샘플
- 채널: 1, 해상도: 128x128
- 값 범위: [0.0, 1.0]
- 분포: Discrete (2 unique): [0.0, 1.0]

### `binary_dataset_128_1.npy`
- Shape: `(100000, 1, 128, 128)` → N=100000 샘플
- 채널: 1, 해상도: 128x128
- 값 범위: [0.0, 1.0]
- 분포: Discrete (2 unique): [0.0, 1.0]

## 스펙트럼 라벨 분석

### `spectra_latest_0.npy`
- Shape: `(100000, 3, 301)` → N=100000 샘플
- 채널: 3 (RGB=3채널 예상), 파장 bin: 301
- 값 범위: [-0.671013, 0.000000]
- Mean: -0.119281, Std: 0.125734

### `spectra_latest_1.npy`
- Shape: `(100000, 3, 301)` → N=100000 샘플
- 채널: 3 (RGB=3채널 예상), 파장 bin: 301
- 값 범위: [-0.633921, 0.000000]
- Mean: -0.107399, Std: 0.124423

## 전체 요약
- 총 구조 이미지 샘플 수: 200000
- 총 스펙트럼 샘플 수: 200000
- 총 디스크 사용량: 3813.9 MB (3.72 GB)
