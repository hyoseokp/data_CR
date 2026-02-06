# Common Context Prompt

## Project Overview
128x128 binary structure 이미지를 입력으로 받아 BGGR 2x2 color filter array의 스펙트럼(30 wavelength bins, 400-700nm)을 예측하는 딥러닝 파이프라인을 구축한다. 핵심 설계 원칙은 두 가지이다: (1) 모델과 loss 함수를 YAML config 한 줄 변경으로 교체 가능한 레지스트리 패턴, (2) 학습 과정을 로컬 브라우저에서 실시간 모니터링하는 WebSocket 기반 대시보드.

데이터는 GitHub 레포(hyoseokp/data_CR)에서 제공되며, 구조 이미지(`binary_dataset_128_*.npy`)와 스펙트럼 라벨(`spectra_latest_*.npy`)로 구성된다. 기본 모델은 MetaSpec_CNNXAttn(CNN + Transformer cross-attention decoder), 기본 loss는 MSE + Pearson Correlation이다.

## File/Directory Convention

### Plan 관련 파일
- plan root: `bot/plans/plan-dl-cr-dashboard/`
- tasks: `tasks/task-XX-<slug>.md` 또는 `tasks/task-XX/` (subtask 포함)
- prompts: `code_gen_prompt/` (Skill 4에서 생성)
- outputs: `code/` (Skill 5에서 생성)

### 프로젝트 코드 구조
```
CR_recon/
├── configs/
│   └── default.yaml              # 모든 하이퍼파라미터 중앙 관리
├── data/
│   ├── __init__.py
│   ├── dataset.py                # Dataset, DataLoader, 전처리
│   ├── binary_dataset_128_0.npy  # 구조 이미지 (다운로드)
│   ├── binary_dataset_128_1.npy
│   ├── spectra_latest_0.npy      # 스펙트럼 라벨 (다운로드)
│   └── spectra_latest_1.npy
├── models/
│   ├── __init__.py               # 모델 레지스트리 (get_model)
│   ├── cnn_xattn.py              # MetaSpec_CNNXAttn
│   └── cnn_gru.py                # MetaSpec_CNNGRU
├── losses/
│   ├── __init__.py               # Loss 레지스트리 (get_loss)
│   ├── mse_pearson.py            # MSE + Pearson Correlation
│   └── weighted_smooth.py        # Weighted MSE + Smoothness
├── dashboard/
│   ├── __init__.py
│   ├── server.py                 # FastAPI + WebSocket 서버
│   ├── hook.py                   # Trainer callback
│   └── static/
│       └── index.html            # 단일 파일 프론트엔드
├── trainer.py                    # Trainer 클래스
├── train.py                      # CLI 진입점
└── requirements.txt
```

## Global Rules

### 코딩 스타일
- Language: Python 3.9+
- Deep Learning: PyTorch (torch, torch.nn, torch.optim)
- Config: PyYAML (YAML 파일 파싱)
- Dashboard server: FastAPI + uvicorn
- Dashboard frontend: 단일 HTML + Chart.js (CDN)
- 타입 힌트 권장하나 강제하지 않음
- 주석은 한국어 허용, 변수명/함수명은 영어

### 모듈화 원칙
- 모든 하이퍼파라미터는 `configs/default.yaml`에 집중 → 코드 내 하드코딩 금지
- 모델 추가: `models/` 아래 파일 작성 → `__init__.py`에 import 한 줄 추가
- Loss 추가: `losses/` 아래 파일 작성 → `__init__.py`에 import 한 줄 추가
- 레지스트리 조회: `get_model(name, **params)`, `get_loss(name, **params)`

### 데이터 규약
- 구조 입력: `(B, 1, 128, 128)` float32, pm1 매핑 시 [-1, 1]
- 스펙트럼 출력: `(B, 2, 2, 30)` float32 (BGGR 배치: [0,0]=B, [0,1]=G, [1,0]=G, [1,1]=R)
- 원본 스펙트럼: (N, 3, 301) RGB → BGGR 변환 후 301→30 bin downsample

### 에러 처리
- 대시보드 서버 미실행 시에도 학습은 정상 진행 (graceful fallback)
- WebSocket push 실패 시 try-except로 무시, 학습 중단 금지
- 존재하지 않는 모델/loss 이름 호출 시 `KeyError` + 등록된 이름 목록 출력

### 재현성
- random seed 고정 (torch, numpy), config에서 설정
- train/val split은 seed 기반 `random_split`으로 재현 가능
- 파일 기반 재현성: 모든 출력은 입력 파일만 근거로 생성

## Tech Assumptions
- Language: Python 3.9+
- Runtime: CUDA GPU (torch.cuda), Windows 10/11
- Package manager: conda / pip
- Core dependencies: torch, numpy, fastapi, uvicorn, websockets, pyyaml, tqdm, matplotlib (fallback)
- Frontend: 빌드 도구 없음, CDN으로 Chart.js 로드

## Config Schema (default.yaml 기준)
```yaml
seed: 42

data:
  struct_files: ["data/binary_dataset_128_0.npy", "data/binary_dataset_128_1.npy"]
  spectra_files: ["data/spectra_latest_0.npy", "data/spectra_latest_1.npy"]
  out_len: 30
  map_to_pm1: true
  augment_180: true
  train_ratio: 0.95
  batch_size: 64
  num_workers: 0

model:
  name: "cnn_xattn"
  params:
    out_len: 30
    d_model: 256
    nhead: 8
    dec_layers: 4
    cnn_dropout: 0.05
    tr_dropout: 0.1
    head_dropout: 0.2

loss:
  name: "mse_pearson"
  params:
    w_mse: 1.0
    w_corr: 0.2

training:
  epochs: 300
  lr: 0.001
  weight_decay: 0.005
  grad_clip: 1.0
  use_amp: true
  warmup_ratio: 0.05
  save_every: 10

dashboard:
  enabled: true
  port: 8501

output:
  dir: "outputs/"
  log_file: "train_log.txt"
```

## WebSocket 메시지 규약
```json
{
  "epoch": 10,
  "total_epochs": 300,
  "lr": 0.00032,
  "train_loss": 0.0123,
  "val_loss": 0.0145,
  "best_val": 0.0098,
  "train_losses": [],
  "val_losses": [],
  "sample": {
    "input_struct": "<base64 PNG or flat 128x128 list>",
    "gt_bggr": [[[2x2x30]]],
    "pred_bggr": [[[2x2x30]]],
    "waves": [400.0, 410.3, ...]
  }
}
```

## Dashboard 시각화 패널 (6개)
1. **Loss 곡선** — train/val, log scale
2. **GT vs Pred 스펙트럼** — B/G/R 각각 GT(점선) vs Pred(실선)
3. **Input Structure** — 128x128 grayscale canvas
4. **GT BGGR 2x2 mean** — lambda 축 평균 후 확대 이미지
5. **Pred BGGR 2x2 mean** — lambda 축 평균 후 확대 이미지
6. **R/G/B Abs Error** — 채널별 |pred - gt| 곡선

## Task 실행 순서
```
Task 01 (데이터 다운로드)
  → Task 02 (프로젝트 구조 + config)
    → Task 03 (데이터 모듈)
    → Task 04 (모델/loss 레지스트리)
      → Task 06 (trainer)
      → Task 05-01 (대시보드 서버)
      → Task 05-02 (대시보드 프론트엔드)
        → Task 05-03 (trainer ↔ 대시보드 연결)
          → Task 07 (통합 테스트)
```

## Definition of Done (global)
- [ ] `python train.py --config configs/default.yaml` 한 줄로 학습+대시보드 실행
- [ ] 브라우저(localhost:8501)에서 6개 패널 실시간 갱신
- [ ] config의 `model.name` / `loss.name` 변경만으로 모델/loss 교체 동작
- [ ] 대시보드 없이도 학습 단독 실행 가능
- [ ] checkpoint resume 지원
