# Plan: dl-cr-dashboard

## User request
- https://github.com/hyoseokp/data_CR 에 올라온 데이터로 딥러닝 돌리고, 실시간으로 로컬 브라우저에서 학습 과정 시각화(loss 그래프, GT vs prediction 비교, input structure, output 2x2 mean, R/G/B별 abs error). 뉴럴넷 모델과 loss를 쉽게 바꿀 수 있도록 모듈화된 구조로 설계.

## Data source
- GitHub: hyoseokp/data_CR
  - `binary_dataset_128_0.npy`, `binary_dataset_128_1.npy` (128x128 구조 이미지)
  - `spectra_latest_0.npy`, `spectra_latest_1.npy` (스펙트럼 라벨)
  - `snapshots/` (스냅샷 데이터)

## Steps
1. GitHub 레포에서 데이터(.npy 파일들)를 로컬에 다운로드하고, 데이터 shape/분포를 분석한다.
2. 프로젝트 디렉토리 구조를 모듈화하여 설계한다 (models/, losses/, data/, dashboard/, configs/).
3. 데이터 로딩 및 전처리 모듈을 만든다 (mmap 로딩, BGGR 변환, binning, 180도 augmentation, train/val split).
4. 모델 레지스트리 모듈을 만들어 모델을 config 한 줄로 교체할 수 있게 한다 (기본: MetaSpec_CNNXAttn).
5. Loss 레지스트리 모듈을 만들어 loss 함수를 config 한 줄로 교체할 수 있게 한다 (기본: MSE + Pearson Correlation).
6. 학습 엔진(trainer)을 구현한다 (AMP, gradient clipping, scheduler, checkpoint 저장).
7. 실시간 대시보드 서버를 구현한다 (Flask/FastAPI + WebSocket으로 브라우저에 학습 상태 스트리밍).
8. 대시보드 프론트엔드를 구현한다 (loss 곡선, GT vs Pred 스펙트럼, input structure 이미지, output 2x2 mean 이미지, R/G/B별 abs error 그래프).
9. trainer와 대시보드를 연결하여 매 epoch 끝에 시각화 데이터를 WebSocket으로 push한다.
10. 전체 파이프라인을 통합 테스트하고, config 파일 하나로 모델/loss/하이퍼파라미터를 변경하여 실행할 수 있는지 검증한다.
