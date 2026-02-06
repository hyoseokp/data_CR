# CR_recon 코드 카탈로그

**목적**: 128×128 구조 이미지 → BGGR 2×2 스펙트럼(30 bins) 예측 딥러닝 시스템

**주요 특징**:
- 모듈화된 아키텍처 (모델/손실/데이터 분리)
- 실시간 대시보드 (WebSocket)
- 180도 회전 데이터 증강
- 배치 크기: 400, GPU: 12GB

---

## 📁 폴더 구조

```
CR_recon/
├── configs/                    # 설정 파일들
│   ├── default.yaml           # 기본 설정 (CNN_XAttn + MSE_Pearson, batch=400)
│   ├── default_no_dashboard.yaml
│   ├── default_weighted.yaml
│   └── test_cnn_gru.yaml
├── data/                       # 데이터 로딩 & 전처리
│   ├── __init__.py
│   ├── dataset.py             # CRDataset: 180도 증강 포함
│   ├── analyze_data.py        # 데이터 분석 도구
│   └── data_summary.md
├── models/                     # 신경망 모델들
│   ├── __init__.py            # 모델 레지스트리
│   ├── cnn_xattn.py           # CNN + Transformer Decoder (현재 사용)
│   └── cnn_gru.py             # CNN + GRU (비교 모델)
├── losses/                     # 손실 함수들
│   ├── __init__.py            # 손실 함수 레지스트리
│   ├── mse_pearson.py         # MSE + Pearson correlation (현재 사용)
│   └── weighted_smooth.py     # MSE + smoothness 정규화
├── dashboard/                  # 실시간 학습 대시보드 (FastAPI + WebSocket)
│   ├── __init__.py
│   ├── server.py              # FastAPI 서버, WebSocket 관리
│   ├── hook.py                # Trainer callback
│   └── static/                # 프론트엔드 (index.html, CSS, JS)
├── train.py                    # CLI 진입점
├── trainer.py                  # 학습 엔진 (메인 루프, 체크포인트)
├── utils.py                    # 유틸리티 (config 로드 등)
├── optimize_hyperparams.py     # 훈련 로그 분석 & 기본 제안
├── SKILL_MODEL_OPTIMIZER.md    # 지능형 모델 최적화 스킬 정의
└── CATALOG.md                  # 이 파일

```

---

## 📄 파일 설명

### 🎯 **메인 진입점**

#### `train.py` ⭐
```
목적: CLI 진입점 + 자동 데이터 정제
사용: python train.py --config configs/default.yaml [--resume checkpoint.pt]

기능:
  1. Config 파일 파싱
  2. 정제된 데이터 확인:
     - 있으면: dataset/bayer/*.npy 로드 (매우 빠름) ✓
     - 없으면: preprocess_data.py 자동 실행
  3. Trainer 인스턴스 생성
  4. 훈련 시작

자동 정제 기능:
  - 함수: ensure_preprocessed_data(cfg_dir)
  - 경로: CR_recon/dataset/bayer/ (수정됨: data_CR-main → dataset)
  - 역할:
    * dataset/bayer/struct_*.npy, bayer_*.npy 존재 확인
    * 없으면 subprocess로 preprocess_data.py 실행
    * 타임아웃: 30분
    * 실패 시: 오류 메시지 출력 후 종료

워크플로우:
  python train.py --config configs/default.yaml
    ↓
  정제 데이터 확인 (dataset/bayer/)
    ├─ 있음: mmap 로드 → 바로 학습 (초 단위)
    └─ 없음: preprocess_data.py 실행
      ├─ 성공: 자동 로드 → 학습
      └─ 실패: 오류 종료

현황:
  ✅ 정제된 데이터 파일 6개 생성됨 (총 1.7 GB)
  ✅ dataset/bayer/ 경로 확인됨
  ✅ train.py에서 정상 감지 가능
```

#### `utils.py`
```
목적: 유틸리티 함수들
주요 기능:
  - load_config(): YAML 설정 파일 로드
  - 경로 처리
```

---

### 🧠 **핵심: 학습 엔진**

#### `trainer.py` ⭐
```
목적: 훈련 루프 및 상태 관리
클래스: Trainer
주요 메서드:
  - __init__(): 모델/loss/데이터로더 초기화
  - train(): 전체 훈련 루프 (epoch 반복)
  - train_one_epoch(): 한 epoch 훈련
  - validate(): 검증
  - save_checkpoint(): 체크포인트 저장
  - load_checkpoint(): 체크포인트 로드
  - log(): 로그 파일에 기록 (epoch마다 flush)

특징:
  - 대시보드 통합 (WebSocket 전송)
  - Callback 시스템
  - AMP (Automatic Mixed Precision) 지원
  - Gradient clipping
  - Cosine annealing + warmup 스케줄러
  - 실시간 로그 저장 (버퍼링 최소화)

로그 형식:
  [EPOCH] N/total_epochs train_loss=X val_loss=Y best_val=Z lr=A
```

---

### 🎨 **모델 아키텍처**

#### `models/__init__.py`
```
목적: 모델 팩토리
함수: get_model(name, **params) → 모델 인스턴스
레지스트리:
  - "cnn_xattn": MetaSpec_CNNXAttn
  - "cnn_gru": MetaSpec_CNNGRU
```

#### `models/cnn_xattn.py` ⭐ (현재 사용)
```
목적: CNN backbone + Transformer decoder
모델: MetaSpec_CNNXAttn
구조:
  1. Stem: 128×128 → 64×64 (5×5 conv)
  2. 4 Stages: CNN residual blocks (stride 2 for downsampling)
     - 각 stage: 2개 residual blocks + circular padding
     - Channels: 64→96→128→192→256
  3. Global pooling: (256, 4, 4) → 256D 벡터
  4. Transformer decoder:
     - Self-attention layers (8 heads, 4 layers)
     - 스펙트럼 시퀀스 생성 (30 bins)
  5. Output: (B, 2, 2, 30) BGGR spectrum

특징:
  - Circular padding (대각선 대칭 구조 반영)
  - GroupNorm + SiLU activation
  - Dropout 정규화
  - Positional encoding for Transformer

입출력:
  입력: (B, 1, 128, 128)
  출력: (B, 2, 2, 30)
```

#### `models/cnn_gru.py`
```
목적: CNN backbone + GRU (경량 비교 모델)
구조:
  1. CNN backbone (CNN_XAttn과 동일)
  2. GRU 레이어 (2 layers)
  3. Linear head (4 bins 예측)

특징:
  - CNN_XAttn보다 빠름
  - 메모리 효율
  - 성능은 CNN_XAttn이 더 좋음
```

---

### 💔 **손실 함수**

#### `losses/__init__.py`
```
목적: 손실 함수 팩토리
함수: get_loss(name, **params) → loss_fn
레지스트리:
  - "mse_pearson": get_mse_pearson_loss
  - "weighted_smooth": get_weighted_smooth_loss
```

#### `losses/mse_pearson.py` ⭐ (현재 사용)
```
목적: MSE + Pearson 상관계수
공식: L = w_mse * MSE(pred, target) + w_corr * (1 - Pearson)
특징:
  - MSE: 절대값 오차
  - Pearson: 스펙트럼 형태 유지 (스케일/시프트 무변)
  - 현재 가중치: w_mse=1.0, w_corr=0.2

장점:
  - 실측값과의 수치적 유사성
  - 스펙트럼 형태(패턴) 일관성
  - 스케일 변화에 강건
```

#### `losses/weighted_smooth.py`
```
목적: MSE + 평활성 정규화
공식: L = w_mse * MSE + w_smooth * smoothness_penalty
특징:
  - 인접 bin 간 차이 최소화
  - 물리적으로 매끄러운 스펙트럼
```

---

### 📊 **데이터 로딩 & 전처리**

#### `data/__init__.py`
```
목적: 데이터로더 팩토리
함수: create_dataloaders(cfg) → (train_loader, val_loader)
```

#### `data/dataset.py` ⭐
```
클래스: CRDataset
목적:
  1. 정제된 Bayer 데이터 또는 원본 spectra 로드
  2. 정제된 데이터 없으면 즉시 정제 (필터링 + 변환)
  3. 180도 회전 데이터 증강
  4. 배치 제공

데이터 처리 (우선순위):
  1. 정제된 데이터 확인:
     - bayer/struct_*.npy, bayer/bayer_*.npy 존재?
     - YES: 바로 로드 → 매우 빠름 ✓
     - NO: 원본에서 즉시 정제 ↓

  2. 정제 (필요시):
     - struct: (N, 1, 128, 128) uint8 → (M, 1, 128, 128) [0,255]
     - spectra: (N, 3, 301) float32 → (M, 2, 2, 30) Bayer
       * 필터링: 0-패딩 샘플 제외
       * 변환: RGB → Bayer [[R,G],[G,B]]
       * 다운샘플: 301 bins → 30 bins

  3. 180도 증강:
     - 원본 M개 + 회전 M개 = 2M개 (augment=True)
     - 회전: R↔B 교환, struct flip

  4. __getitem에서:
     - struct: [-1, 1] 정규화 (map_to_pm1=True)
     - spectrum: 부호 반전 (음수 → 양수)

출력:
  - struct: (1, 128, 128) float32 ∈ [-1, 1]
  - spectrum: (2, 2, 30) float32 Bayer pattern
    * [0,0]=R, [0,1]=G, [1,0]=G, [1,1]=B

매개변수:
  - augment_180: bool (증강 활성화)
  - out_len: int (출력 bins, 기본 30)
  - map_to_pm1: bool (정규화, 기본 True)

성능:
  - 정제 데이터 있음: 초 단위 로드 (mmap)
  - 정제 데이터 없음: 첫 실행만 ~30초 정제
```

#### `preprocess_data.py` ⭐
```
목적: 원본 데이터를 Bayer 패턴으로 전처리하여 저장
사용: python preprocess_data.py (자동 실행 or 수동 실행)

기능:
  1. 원본 파일 로드
     - struct: (N, 1, 128, 128) from data_CR-main/
     - spectra: (N, 3, 301) from data_CR-main/

  2. 유효 샘플 필터링
     - 0-패딩 샘플 제외 (200,000 → 105,601)

  3. Bayer 패턴 변환
     - (3, 301) → (2, 2, 30)
     - 레이아웃: [[R,G],[G,B]] (중요: R≠B, G1=G2)
     - 다운샘플: 301 bins → 30 bins

  4. 180도 회전 생성
     - R↔B 교환 (대각선 대칭 구조 반영)

  5. 저장 (dataset/bayer/) ✅
     - struct_0.npy (859 MB), struct_1.npy (791 MB)
     - bayer_0.npy (25 MB), bayer_1.npy (23 MB)
     - bayer_rotated_0.npy (25 MB), bayer_rotated_1.npy (23 MB)

성능:
  - 출력: 약 1.7 GB (struct 1.65 GB + bayer 0.05 GB)
  - 시간: 약 4-5초 (전체 정제)
  - 로드: ~100ms (mmap)

통합:
  - train.py 실행 시 자동 호출 (ensure_preprocessed_data)
  - 정제된 데이터 있으면: 재실행 스킵
  - 정제된 데이터 없으면: subprocess 실행 (타임아웃 30분)
  - 수동 실행도 가능: python preprocess_data.py

최신 상태: ✅ 완료
  - 6개 파일 성공 생성
  - train.py 경로 동기화됨
  - dataset.py 호환됨
```

#### `data/analyze_data.py`
```
목적: 데이터셋 분석 도구
기능:
  - 데이터 크기, 범위 확인
  - 샘플 시각화
  - 통계 계산
```

---

### 📺 **대시보드 (실시간 시각화)**

#### `dashboard/server.py` ⭐
```
목적: FastAPI + WebSocket 대시보드 서버
클래스: DashboardServer
포트: 8501 (기본값)

주요 메서드:
  - start(): 서버 시작 (별도 스레드)
  - stop(): 서버 종료
  - push_update(): epoch 결과 전송
  - push_progress(): batch 진행 상황 전송
  - reset_state(): 훈련 시작 시 상태 초기화

상태 (self.state):
  - epoch, total_epochs, lr
  - train_loss, val_loss, best_val
  - train_losses, val_losses (히스토리)
  - progress (stage, batch, total_batches, current_loss)
  - sample (선택사항)

WebSocket:
  - 클라이언트 연결 시 현재 상태 전송
  - epoch/batch 완료 시 업데이트
```

#### `dashboard/hook.py`
```
목적: Trainer callback
함수: DashboardHook(trainer)
기능:
  - epoch 완료 후 메트릭 업데이트
  - 대시보드에 전송
```

#### `dashboard/static/index.html`
```
목적: 프론트엔드 (브라우저 시각화)
기능:
  - 손실 그래프 (Chart.js)
  - 입력 이미지 시각화 (Canvas)
  - GT vs 예측 스펙트럼 비교
  - Epoch/Batch 진행 바
  - 모델/손실 정보 (LaTeX with KaTeX)
  - LocalStorage로 새로고침 후에도 데이터 유지
```

---

### 🔧 **훈련 로그 분석 & 최적화**

#### `optimize_hyperparams.py`
```
목적: 훈련 로그 분석 후 기본 개선 제안
함수:
  - parse_train_log(): 로그 파싱
  - analyze_performance(): 지표 계산
  - generate_suggestions(): 규칙 기반 제안

제안 범위:
  - Learning rate 조정
  - 모델 전환 (CNN_XAttn ↔ CNN_GRU)
  - 손실 함수 전환
  - Weight decay, batch size 조정

사용: python optimize_hyperparams.py --log outputs/train_log.txt
```

#### `SKILL_MODEL_OPTIMIZER.md` ⭐
```
목적: 지능형 모델 최적화 스킬 정의
사용: ## 모델 optimizing 해줘 (또는 /model-optimizer)

프로세스:
  1. 모든 train_log.txt 수집
  2. 성능 지표 계산 (개선율, 수렴성, 과적합도)
  3. 패턴 인식 (어떤 설정이 효과적인가)
  4. 근본 원인 분석 (왜 개선 정체됨?)
  5. 창의적 제안 (새로운 아이디어)
  6. 최고 전략 추천
  7. 실행 (동의 시)

제안 범위:
  - 새로운 모델 아키텍처
  - 새로운 손실 함수
  - 하이퍼파라미터 조정
  - 조합 전략
```

---

### ⚙️ **설정 파일**

#### `configs/default.yaml` ⭐ (현재 활용)
```yaml
# 데이터
data:
  struct_files: [binary_dataset_128_0.npy, binary_dataset_128_1.npy]
  spectra_files: [spectra_latest_0.npy, spectra_latest_1.npy]
  out_len: 30
  batch_size: 400            # ← 최적화됨 (원래 64)
  augment_180: true          # ← 180도 회전 증강
  train_ratio: 0.95

# 모델
model:
  name: cnn_xattn
  params:
    out_len: 30
    d_model: 256
    nhead: 8
    dec_layers: 4
    cnn_dropout: 0.05
    tr_dropout: 0.1
    head_dropout: 0.2
    use_circular_padding: true

# 손실
loss:
  name: mse_pearson
  params:
    w_mse: 1.0
    w_corr: 0.2

# 훈련
training:
  epochs: 300
  lr: 0.001
  weight_decay: 0.005
  grad_clip: 1.0
  use_amp: true             # Automatic Mixed Precision
  warmup_ratio: 0.05
  save_every: 10

# 대시보드
dashboard:
  enabled: true
  port: 8501

output:
  dir: outputs/
  log_file: train_log.txt
```

#### `configs/test_cnn_gru.yaml`
```
CNN_GRU 모델로 비교 테스트용
```

#### `configs/default_weighted.yaml`
```
Weighted_Smooth 손실함수 테스트용
```

#### `configs/default_no_dashboard.yaml`
```
대시보드 없이 훈련 (속도 테스트)
```

---

## 🚀 **사용 방법**

### 1. 기본 훈련
```bash
cd CR_recon
python train.py --config configs/default.yaml
```

### 2. 중단된 훈련 재개
```bash
python train.py --config configs/default.yaml --resume outputs/cnn_xattn_best.pt
```

### 3. 다른 모델로 테스트
```bash
python train.py --config configs/test_cnn_gru.yaml
```

### 4. 훈련 로그 분석
```bash
python optimize_hyperparams.py --log outputs/train_log.txt
```

### 5. 지능형 모델 최적화
```
## 모델 optimizing 해줘
```

### 6. 대시보드 접속
```
http://localhost:8501
```

---

## 📈 **현재 성능**

```
Configuration: CNN_XAttn + MSE_Pearson + batch=400
Data: 200,640 train (180도 증강) + 5,281 val
GPU: NVIDIA 12GB

Epoch 5:
  - best_val_loss: 0.0781
  - train_loss: 0.0838
  - 55% 개선 (from epoch 1: 0.174)

Est. Time:
  - 한 epoch: ~2.5분
  - 300 epochs: ~12.5시간
```

---

## 🔄 **확장 가능성**

### 쉽게 추가 가능한 항목
1. **새 모델**: `models/my_model.py` + `models/__init__.py`에 등록
2. **새 손실**: `losses/my_loss.py` + `losses/__init__.py`에 등록
3. **새 설정**: `configs/test_xxx.yaml` 생성

### 구현 예
```python
# 1. 새 모델 구현
class MetaSpec_MyModel(nn.Module):
    def __init__(self, out_len=30, **params):
        super().__init__()
        # ... 구현 ...

    def forward(self, x):
        # 입력: (B, 1, 128, 128)
        # 출력: (B, 2, 2, out_len)
        return output

# 2. 레지스트리에 등록
# models/__init__.py에 추가:
from .my_model import MetaSpec_MyModel
_MODELS["my_model"] = MetaSpec_MyModel
```

---

## 📝 **주요 개선 사항**

| 항목 | 상태 |
|------|------|
| 180도 회전 증강 | ✅ 구현됨 (로드 시점) |
| 배치 사이즈 최적화 | ✅ 400으로 설정 |
| 로그 즉시 저장 | ✅ flush + fsync |
| 대시보드 상태 초기화 | ✅ 훈련 시작 시 |
| Circular padding | ✅ 대각선 대칭 반영 |
| 데이터 정제 | ✅ 0-패딩 샘플 제외 |
| 모델 최적화 스킬 | ✅ SKILL_MODEL_OPTIMIZER.md |

---

**마지막 업데이트**: 2026-02-06
**상태**: 운영 중 (300 epochs 훈련 진행)

