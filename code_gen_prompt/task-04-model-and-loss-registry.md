# Coding Prompt: Task-04 모델 레지스트리 + Loss 레지스트리

## Purpose
모델과 loss 함수를 문자열 이름으로 등록/조회할 수 있는 레지스트리 패턴을 구현하여, config 한 줄 변경만으로 모델과 loss를 교체할 수 있게 한다.

## Context
- 프로젝트는 `bot/plans/plan-dl-cr-dashboard/` 안에서 진행한다.
- Language: Python 3.9+, PyTorch
- 기존 모델 구현:
  - MetaSpec_CNNXAttn: CNN (5 stages) + Transformer decoder with cross-attention
    - 입력: (B, 1, 128, 128) → CNN backbone → (B, C, 4, 4) spatial
    - Transformer: self-attn on wavelength tokens + cross-attn to spatial tokens
    - 출력: (B, 2, 2, 30)
    - 핵심 파라미터: out_len=30, d_model=256, nhead=8, dec_layers=4, cnn_dropout=0.05, tr_dropout=0.1, head_dropout=0.2
  - MetaSpec_CNNGRU: CNN + GRU (비교용)
- 기존 loss 구현:
  - MSE + Pearson Correlation: w_mse=1.0, w_corr=0.2
  - Weighted MSE + Smoothness: 가중치 각각 다름

## Task Description
1. `models/cnn_xattn.py`에서 MetaSpec_CNNXAttn을 정의한다.
   - ResBlockGN, XAttnDecoderBlock 등 필요한 helper 클래스 포함
   - config의 model.params로 전달받은 파라미터로 초기화

2. `models/cnn_gru.py`에서 MetaSpec_CNNGRU을 정의한다.
   - 기존 notebook의 CNNGRU 구현 참조

3. `models/__init__.py`에서 모델 레지스트리 구현
   - dict 기반 레지스트리: `_MODELS = {"cnn_xattn": MetaSpec_CNNXAttn, "cnn_gru": MetaSpec_CNNGRU}`
   - `get_model(name, **kwargs) → nn.Module` 함수
   - 존재하지 않는 모델명 호출 시: "KeyError: Unknown model 'xxx'. Available: [cnn_xattn, cnn_gru]" 에러 메시지

4. `losses/mse_pearson.py`에서 MSE + Pearson Correlation loss 구현
   - config에서 `w_mse`, `w_corr` 파라미터를 받음
   - Pearson correlation: per-sample, per-channel, along wavelength dimension
   - 수식: r = (pred - pred_mean) · (tgt - tgt_mean) / (||pred-pred_mean|| · ||tgt-tgt_mean|| + eps)
   - Loss = w_mse * MSE + w_corr * (1 - r).mean()

5. `losses/weighted_smooth.py`에서 Weighted MSE + Smoothness loss 구현
   - notebook의 Cell 2 loss 참조
   - 가중치 기반 MSE + temporal smoothness 항

6. `losses/__init__.py`에서 loss 레지스트리 구현
   - dict 기반 레지스트리: `_LOSSES = {"mse_pearson": ..., "weighted_smooth": ...}`
   - `get_loss(name, **kwargs) → callable` 함수
   - 반환값은 (pred, target) → scalar loss를 계산하는 callable
   - 존재하지 않는 loss명 호출 시: "KeyError: Unknown loss 'xxx'. Available: [mse_pearson, weighted_smooth]" 에러 메시지

## Inputs
- `bot/plans/plan-dl-cr-dashboard/common_context_prompt.md` (모델/loss 파라미터 스키마)
- `C:\Users\연구실\Research_CR_NA .ipynb` (MetaSpec_CNNXAttn, MetaSpec_CNNGRU, loss 구현 참조)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/configs/default.yaml` (model.params, loss.params)

## Outputs
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/models/__init__.py` (레지스트리)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/models/cnn_xattn.py` (MetaSpec_CNNXAttn)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/models/cnn_gru.py` (MetaSpec_CNNGRU)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/losses/__init__.py` (레지스트리)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/losses/mse_pearson.py` (MSE+Pearson)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/losses/weighted_smooth.py` (Weighted+Smooth)

## Constraints
- 레지스트리는 dict 기반 (간단하고 명확)
- `get_model(name, **kwargs)` / `get_loss(name, **kwargs)` 함수 필수
- config의 model.params / loss.params로 전달받은 딕셔너리를 **kwargs로 함수에 전달
- 모델 파라미터 기본값은 common_context_prompt.md의 Config Schema 참조
- Loss callable은 `loss_fn(pred, target)` 형태로 호출 가능해야 함
- 코드 내 하드코딩 금지, 모든 파라미터는 config에서 받음

## Implementation Rules
- Python 3.9+ 호환
- PyTorch nn.Module 상속
- 주석은 한국어 허용, 변수명/함수명은 영어
- 모델 입출력:
  - 입력: (B, 1, 128, 128) float32
  - 출력: (B, 2, 2, 30) float32
- Loss 입출력: (pred, target) → scalar (float)

## Expected Result / Acceptance Criteria
- [ ] `get_model("cnn_xattn", out_len=30, d_model=256, ...)` 호출 시 MetaSpec_CNNXAttn 인스턴스 반환
- [ ] `get_model("cnn_gru", ...)` 호출 시 MetaSpec_CNNGRU 인스턴스 반환
- [ ] 존재하지 않는 모델명 호출 시 "Unknown model" KeyError 메시지
- [ ] `get_loss("mse_pearson", w_mse=1.0, w_corr=0.2)` 호출 시 callable 반환
- [ ] `get_loss("weighted_smooth", ...)` 호출 시 callable 반환
- [ ] 손실함수 callable: `loss_value = loss_fn(pred, target)` 형태로 호출 가능
- [ ] 존재하지 않는 loss명 호출 시 "Unknown loss" KeyError 메시지
