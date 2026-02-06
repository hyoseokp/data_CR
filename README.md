# CR_DL_auto

딥러닝 기반 분광 측정 자동화 시스템 (Deep Learning-based Spectral Measurement Automation)

## 📋 프로젝트 개요

구조 이미지(128×128)로부터 분광 특성(BGGR 2×2, 30 wavelength bins)을 예측하는 딥러닝 기반 자동화 시스템입니다.

**특징:**
- 모듈화된 설계: 모델과 손실함수를 config로 쉽게 변경 가능
- 실시간 대시보드: WebSocket 기반 학습 과정 시각화
- 원형 주기적 구조 학습: Circular padding으로 periodic boundary condition 반영
- AMP & 최적화: Mixed Precision, Cosine Annealing with Warmup, Gradient Clipping

## 🏗️ 프로젝트 구조

```
CR_recon/
├── configs/                    # 학습 설정 파일
│   ├── default.yaml           # 기본 설정 (MetaSpec_CNNXAttn + MSE_Pearson)
│   ├── test_cnn_gru.yaml      # GRU 모델 테스트
│   ├── default_weighted.yaml  # Weighted Smooth Loss
│   └── default_no_dashboard.yaml
├── data/
│   ├── dataset.py             # 데이터 로딩, RGB→BGGR 변환, 180° augmentation
│   └── download_data.py       # 데이터 다운로드
├── models/
│   ├── cnn_xattn.py           # MetaSpec_CNNXAttn: CNN + Transformer Decoder
│   ├── cnn_gru.py             # MetaSpec_CNNGRU: CNN + GRU (baseline)
│   └── __init__.py            # 모델 registry
├── losses/
│   ├── mse_pearson.py         # MSE + Pearson Correlation
│   ├── weighted_smooth.py     # Weighted MSE + Smoothness Regularization
│   └── __init__.py            # Loss registry
├── dashboard/
│   ├── server.py              # FastAPI + WebSocket 서버
│   ├── hook.py                # Trainer callback (대시보드 데이터 수집)
│   └── static/
│       └── index.html         # 대시보드 프론트엔드 (Chart.js + KaTeX)
├── trainer.py                 # 학습 엔진 (AMP, 스케줄러, 콜백)
├── train.py                   # 학습 실행 스크립트
└── requirements.txt           # 의존성
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
python CR_recon/data/download_data.py
# 또는 수동으로 데이터를 data_CR-main/ 디렉토리에 배치
```

### 3. 학습 실행

```bash
# 기본 설정으로 학습
python CR_recon/train.py

# 커스텀 config 사용
python CR_recon/train.py --config CR_recon/configs/test_cnn_gru.yaml
```

### 4. 대시보드 확인

학습 시작 후 브라우저에서 열기:
```
http://localhost:8501
```

## 📊 대시보드 기능

**실시간 시각화:**
- **Epoch/Batch Progress Bars**: 학습 진행률 표시
- **Loss Curves**: Train/Val loss (log scale)
- **GT vs Prediction**: 분광 데이터 비교
- **Input Structure**: 입력 구조 이미지 시각화
- **BGGR 2×2 Mean**: 각 채널별 평균값
- **Abs Error**: 절대 오차 분석
- **Model & Loss Info**: LaTeX로 표시된 모델 아키텍처 및 손실함수

**특징:**
- 자동 재연결 (exponential backoff)
- LocalStorage로 새로고침 시 데이터 복원
- KaTeX 수식 렌더링

## 🔧 모델 및 Loss 함수

### 모델

| 모델 | 설명 |
|-----|------|
| **MetaSpec_CNNXAttn** | 5-stage CNN backbone + Transformer Decoder with Cross-Attention |
| **MetaSpec_CNNGRU** | CNN backbone + GRU Decoder (간단한 baseline) |

### Loss 함수

| 손실함수 | 설명 |
|--------|------|
| **MSE_Pearson** | MSE + Pearson Correlation (scale/shift invariant) |
| **Weighted_Smooth** | Weighted MSE + 1차/2차 미분 정규화 |

## 📝 Config 구조

```yaml
seed: 42

data:
  struct_files: [...]         # 구조 이미지 경로
  spectra_files: [...]        # 분광 데이터 경로
  out_len: 30                 # 출력 wavelength bins
  augment_180: true           # 180° 회전 augmentation
  train_ratio: 0.95

model:
  name: "cnn_xattn"
  params:
    d_model: 256
    use_circular_padding: true  # Circular padding 활성화

loss:
  name: "mse_pearson"
  params:
    w_mse: 1.0
    w_corr: 0.2

training:
  epochs: 300
  lr: 0.001
  use_amp: true               # Mixed Precision
  warmup_ratio: 0.05          # Cosine annealing + warmup

dashboard:
  enabled: true
  port: 8501
```

## 🎯 주요 기술

- **Circular Padding**: 대각선 주기성까지 고려한 2D 원형 패딩
- **Spectral Normalization**: Group Norm으로 안정적인 학습
- **Transformer Decoder**: Cross-attention으로 공간-파장 정보 융합
- **AMP**: FP16으로 메모리 효율성 및 속도 향상
- **WebSocket Dashboard**: 실시간 학습 모니터링

## 📚 데이터 형식

**입력:** (B, 1, 128, 128) float32
- 구조 이미지 (grayscale)
- [-1, 1] 범위로 정규화

**출력:** (B, 2, 2, 30) float32
- BGGR 2×2 배치로 정렬된 분광 데이터
- 30개 wavelength bins (400-700nm)

## 🔬 결과 저장

학습 완료 후 outputs/ 디렉토리에 저장:
- `{model_name}_best.pt`: 최고 성능 모델
- `{model_name}_last.pt`: 마지막 epoch 모델
- `{model_name}_epoch_XXXX.pt`: 주기적 체크포인트
- `train_log.txt`: 학습 로그

## 📄 라이선스

MIT License

## 👨‍💻 개발자

Created for automated spectral measurement systems
