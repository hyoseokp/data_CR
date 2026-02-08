# MetaSpec CR Reconstruction

128x128 binary structure -> RGB spectra (Bayer pattern) reconstruction using CNN + Cross-Attention model.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
cd code/CR_recon
set PYTHONIOENCODING=utf-8&&C:/anaconda3/python.exe train.py --config configs/default.yaml
```

실행하면 기존 best checkpoint이 있을 경우 자동으로 물어봅니다:
```
[INFO] 기존 best checkpoint 발견: .../cnn_xattn_best.pt
[선택] 기존 best 파라미터를 불러올까요? (y/n):
```
- `y` : best 모델 가중치를 불러온 뒤 epoch 0부터 학습
- `n` : 완전히 새로 학습

## Command Options

| 옵션 | 설명 |
|---|---|
| `--config` | config 파일 경로 (필수) |
| `--resume` | checkpoint에서 학습 이어서 진행 (optimizer/scheduler/epoch 복원) |
| `--init-weights` | checkpoint에서 모델 가중치만 로드 (epoch 0부터 새로 학습) |

```bash
# 새로 학습
python train.py --config configs/default.yaml

# 학습 이어서 진행
python train.py --config configs/default.yaml --resume outputs/cnn_xattn_last.pt

# best 가중치만 불러와서 새로 학습
python train.py --config configs/default.yaml --init-weights outputs/cnn_xattn_best.pt
```

## Auto Features

- **데이터 자동 다운로드**: `data_CR-main/` 데이터가 1일 이상 오래되면 GitHub에서 zip으로 자동 업데이트
- **데이터 자동 전처리**: `dataset/bayer/` 에 정제 데이터 없으면 `preprocess_data.py` 자동 실행
- **대시보드**: 학습 시작 시 `http://localhost:8501` 에서 실시간 loss/prediction 확인

## Project Structure

```
code/CR_recon/
  train.py              # CLI 진입점
  trainer.py            # 학습 엔진
  preprocess_data.py    # 데이터 전처리 (spectra -> Bayer)
  utils.py              # config 로드 등 유틸
  configs/
    default.yaml        # 학습 설정
  models/
    cnn_xattn.py        # CNN + Cross-Attention 모델
    cnn_gru.py          # CNN + GRU 모델
  losses/
    mse_pearson.py      # MSE + Pearson Correlation loss
  dashboard/
    server.py           # FastAPI WebSocket 서버
    hook.py             # Trainer callback
    static/             # Dashboard frontend
  dataset/bayer/        # 전처리된 데이터 (자동 생성)
  outputs/              # checkpoint, 로그
data_CR-main/           # 원본 데이터 (자동 다운로드)
```

## Config (default.yaml)

| 항목 | 기본값 | 설명 |
|---|---|---|
| `model.name` | cnn_xattn | 모델 (cnn_xattn / cnn_gru) |
| `training.epochs` | 500 | 총 epoch 수 |
| `training.lr` | 0.001 | 학습률 (cosine decay + linear warmup) |
| `training.warmup_ratio` | 0.05 | warmup 비율 (25 epochs) |
| `loss.params.blue_weight` | 2.0 | Blue 채널 MSE 가중치 |
| `data.batch_size` | 400 | 배치 크기 |
| `data.augment_180` | true | 180도 회전 augmentation |
| `dashboard.port` | 8501 | 대시보드 포트 |
