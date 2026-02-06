# Task 05-03: Trainer ↔ 대시보드 연결

## 목적
trainer의 학습 루프에서 매 epoch 종료 시 시각화에 필요한 데이터를 수집하여 대시보드 서버로 push하는 연결 로직을 구현한다.

## 입력
- `CR_recon/trainer.py` (Task 06에서 구현)
- `CR_recon/dashboard/server.py` (Task 05-01)

## 출력
- `CR_recon/trainer.py` 내 대시보드 callback/hook 추가
- 또는 `CR_recon/dashboard/hook.py` — trainer에 주입할 callback 클래스

## 전송 데이터 (JSON)
```json
{
  "epoch": 10,
  "total_epochs": 300,
  "lr": 0.00032,
  "train_loss": 0.0123,
  "val_loss": 0.0145,
  "best_val": 0.0098,
  "train_losses": [0.05, 0.04, ...],
  "val_losses": [0.06, 0.05, ...],
  "sample": {
    "input_struct": [[...128x128...]],
    "gt_bggr": [[2x2x30]],
    "pred_bggr": [[2x2x30]],
    "waves": [400, 410, ...]
  }
}
```

## 제약
- 이미지 데이터(128x128)는 base64 PNG 또는 flat list로 전송
- 스펙트럼 데이터는 nested list로 전송
- 대시보드 서버 미실행 시에도 학습은 정상 진행 (graceful fallback)
- push 실패 시 학습 중단되지 않도록 try-except 처리

## 완료 조건
- [ ] 매 epoch 종료 시 val sample 1개의 시각화 데이터가 WebSocket으로 전송
- [ ] 대시보드 서버 없이 학습 시작해도 에러 없이 동작
- [ ] 브라우저에서 실시간 갱신 확인
