# Coding Prompt: Task-01 데이터 다운로드 및 분석

## Purpose
GitHub 레포(hyoseokp/data_CR)에서 .npy 데이터 파일 4개를 로컬 `CR_recon/data/` 디렉토리에 다운로드하고, 각 파일의 shape/dtype/값 범위/분포를 분석하여 `data_summary.md`에 기록한다.

## Context
- 이 프로젝트는 128x128 binary structure 이미지 → BGGR 2x2 스펙트럼(30 bins) 예측 딥러닝 파이프라인이다.
- 데이터 소스: https://github.com/hyoseokp/data_CR
- 구조 이미지 파일: `binary_dataset_128_0.npy`, `binary_dataset_128_1.npy`
- 스펙트럼 라벨 파일: `spectra_latest_0.npy`, `spectra_latest_1.npy`
- 프로젝트 루트: `CR_recon/`
- Runtime: Windows 10/11, Python 3.9+
- Git LFS를 사용할 수 있으므로 다운로드 방식을 적절히 선택해야 한다.

## Task Description
1. `CR_recon/data/` 디렉토리가 없으면 생성한다.
2. GitHub 레포에서 4개 .npy 파일을 다운로드한다 (git clone 또는 직접 다운로드).
3. 각 파일에 대해 numpy로 로드하여 다음을 분석한다:
   - shape, dtype
   - min, max, mean, std
   - 값 분포 특성 (binary 0/1인지, continuous인지 등)
4. structure 데이터: 채널 수, 해상도(128x128 확인), 값 범위
5. spectra 데이터: 채널 수(3=RGB?), 파장 bin 수(301?), 값 범위
6. 분석 결과를 `CR_recon/data/data_summary.md`에 기록한다.

## Inputs
- GitHub 레포: `https://github.com/hyoseokp/data_CR`
  - `binary_dataset_128_0.npy`
  - `binary_dataset_128_1.npy`
  - `spectra_latest_0.npy`
  - `spectra_latest_1.npy`

## Outputs
- `CR_recon/data/binary_dataset_128_0.npy` (다운로드된 파일)
- `CR_recon/data/binary_dataset_128_1.npy` (다운로드된 파일)
- `CR_recon/data/spectra_latest_0.npy` (다운로드된 파일)
- `CR_recon/data/spectra_latest_1.npy` (다운로드된 파일)
- `CR_recon/data/data_summary.md` (분석 결과)

## Constraints
- Git LFS 사용 여부를 확인하고 적절한 방법으로 다운로드 (.gitattributes에 LFS 설정 확인)
- 다운로드 전 로컬 디스크 공간이 충분한지 확인
- numpy mmap_mode="r"로 메모리 효율적 로딩 (분석 시)
- 이미 다운로드된 파일이 있으면 재다운로드하지 않음

## Implementation Rules
- Python 스크립트 또는 shell 명령어 조합으로 구현
- `data_summary.md`는 마크다운 테이블 형식으로 작성
- 분석 스크립트는 `CR_recon/data/analyze_data.py`로 저장 (재실행 가능)

## Expected Result / Acceptance Criteria
- [ ] 4개 .npy 파일이 `CR_recon/data/`에 존재
- [ ] 각 파일의 shape, dtype이 data_summary.md에 기록
- [ ] structure 데이터: 값 범위(0/1 binary vs continuous), 채널 수, 해상도 확인됨
- [ ] spectra 데이터: 채널 수(RGB 3채널?), 파장 bin 수(301?), 값 범위 확인됨
- [ ] data_summary.md가 마크다운 형식으로 정리됨
