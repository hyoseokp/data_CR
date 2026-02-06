# Task 04: 모델 레지스트리 + Loss 레지스트리

## 목적
모델과 loss 함수를 문자열 이름으로 등록/조회할 수 있는 레지스트리 패턴을 구현하여, config 한 줄 변경만으로 모델과 loss를 교체할 수 있게 한다.

## 입력
- 기존 모델 코드: MetaSpec_CNNXAttn (ResBlockGN, XAttnDecoderBlock 포함)
- 기존 모델 코드: MetaSpec_CNNGRU (비교용)
- 기존 loss 코드: MSE + Pearson Correlation
- 기존 loss 코드: Weighted MSE + Smoothness (Cell 2의 loss)
- `configs/default.yaml`의 model/loss 섹션

## 출력
- `CR_recon/models/__init__.py` — 모델 레지스트리 (get_model 함수)
- `CR_recon/models/cnn_xattn.py` — MetaSpec_CNNXAttn 정의
- `CR_recon/models/cnn_gru.py` — MetaSpec_CNNGRU 정의
- `CR_recon/losses/__init__.py` — loss 레지스트리 (get_loss 함수)
- `CR_recon/losses/mse_pearson.py` — MSE + Pearson loss
- `CR_recon/losses/weighted_smooth.py` — Weighted MSE + Smoothness loss

## 제약
- 레지스트리는 decorator 기반 (`@register_model("cnn_xattn")`) 또는 dict 기반
- `get_model(name, **kwargs)` → nn.Module 인스턴스 반환
- `get_loss(name, **kwargs)` → loss 함수(callable) 반환
- 새 모델/loss 추가 시 파일 하나 만들고 등록만 하면 되는 구조
- config의 model.params, loss.params로 하이퍼파라미터 전달

## 완료 조건
- [ ] `get_model("cnn_xattn", out_len=30, ...)` 호출 시 MetaSpec_CNNXAttn 반환
- [ ] `get_model("cnn_gru", out_len=30, ...)` 호출 시 MetaSpec_CNNGRU 반환
- [ ] `get_loss("mse_pearson")` 호출 시 MSE+Pearson loss callable 반환
- [ ] `get_loss("weighted_smooth")` 호출 시 Weighted+Smooth loss callable 반환
- [ ] 존재하지 않는 이름 호출 시 명확한 에러 메시지
