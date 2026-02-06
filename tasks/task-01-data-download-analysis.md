# Task 01: 데이터 다운로드 및 분석

## 목적
GitHub 레포(hyoseokp/data_CR)에서 .npy 데이터 파일들을 로컬에 다운로드하고, shape/dtype/값 범위/분포를 파악하여 이후 전처리 설계의 근거를 마련한다.

## 입력
- GitHub 레포: https://github.com/hyoseokp/data_CR
  - `binary_dataset_128_0.npy`, `binary_dataset_128_1.npy`
  - `spectra_latest_0.npy`, `spectra_latest_1.npy`

## 출력
- `CR_recon/data/` 디렉토리에 다운로드된 .npy 파일들
- `CR_recon/data/data_summary.md` (각 파일의 shape, dtype, min/max/mean/std, 샘플 시각화 결과 요약)

## 제약
- Git LFS 사용 여부 확인 후 적절한 방법으로 다운로드
- 로컬 디스크 공간 확인 (파일 크기 사전 체크)
- 파일 기반 재현성 유지

## 완료 조건
- [ ] 4개 .npy 파일이 로컬에 존재
- [ ] 각 파일의 shape, dtype 확인 완료
- [ ] structure 데이터: 값 범위(0/1 binary vs continuous), 채널 수, 해상도 확인
- [ ] spectra 데이터: 채널 수(RGB 3채널?), 파장 bin 수(301?), 값 범위 확인
- [ ] data_summary.md 작성 완료
