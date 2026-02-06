# Coding Prompt: Task-02 프로젝트 구조 설계 및 Config 시스템

## Purpose
모듈화된 프로젝트 디렉토리 구조를 생성하고, YAML config 한 줄 변경으로 모델/loss/하이퍼파라미터를 교체할 수 있는 설정 시스템을 구축한다.

## Context
- 프로젝트는 `bot/plans/plan-dl-cr-dashboard/` 안에서 진행한다.
- 데이터는 `data_CR-main/` 에 이미 다운로드됨 (Task-01 완료).
  - `binary_dataset_128_0.npy`, `binary_dataset_128_1.npy`: (100000,1,128,128) uint8 binary
  - `spectra_latest_0.npy`, `spectra_latest_1.npy`: (100000,3,301) float32, 값 범위 [-0.67, 0]
- Language: Python 3.9+, PyTorch, PyYAML
- 모든 하이퍼파라미터는 config에 집중, 코드 내 하드코딩 금지
- 모델 레지스트리: `get_model(name, **params)`, Loss 레지스트리: `get_loss(name, **params)`

## Task Description
1. 프로젝트 루트 아래에 모듈 디렉토리 구조를 생성한다 (configs/, data/, models/, losses/, dashboard/, dashboard/static/).
2. 각 파이썬 패키지에 빈 `__init__.py`를 배치한다.
3. `configs/default.yaml`에 common_context_prompt.md의 Config Schema를 그대로 반영한다.
   - data 경로는 `data_CR-main/` 기준 상대경로로 설정한다.
4. config 로딩 유틸리티 `utils.py`를 작성한다 (YAML 로드 + dict → namespace 변환).
5. `requirements.txt`를 작성한다.

## Inputs
- `bot/plans/plan-dl-cr-dashboard/common_context_prompt.md` (Config Schema 참조)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/data/data_summary.md` (데이터 사양)

## Outputs
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/configs/default.yaml`
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/data/__init__.py`
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/models/__init__.py`
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/losses/__init__.py`
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/dashboard/__init__.py`
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/dashboard/static/.gitkeep`
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/utils.py`
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/requirements.txt`

## Constraints
- config는 YAML 단일 파일로 관리
- `model.name`, `loss.name` 필드 변경만으로 모델/loss 교체 가능
- 데이터 경로: `../data_CR-main/` (CR_recon 기준 상대경로)
- 기존 Research_CR_NA .ipynb의 하이퍼파라미터를 default 값으로 반영:
  - batch_size=64, lr=1e-3, weight_decay=5e-3, epochs=300, grad_clip=1.0
  - model: cnn_xattn, d_model=256, nhead=8, dec_layers=4
  - loss: mse_pearson, w_mse=1.0, w_corr=0.2

## Implementation Rules
- Python 3.9+ 호환
- `utils.py`의 `load_config(path)` 함수: YAML 파일 → dict 반환
- 주석은 한국어 허용, 변수명/함수명은 영어
- requirements.txt에 주요 의존성 기재: torch, numpy, pyyaml, fastapi, uvicorn, websockets, tqdm

## Expected Result / Acceptance Criteria
- [ ] 디렉토리 구조가 common_context_prompt.md의 프로젝트 코드 구조와 일치
- [ ] default.yaml에 seed, data, model, loss, training, dashboard, output 모든 섹션 포함
- [ ] `load_config(path)` 함수가 YAML을 dict로 로드
- [ ] requirements.txt에 모든 핵심 의존성 기재
- [ ] 모든 패키지 디렉토리에 `__init__.py` 존재
