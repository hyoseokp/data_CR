# Coding Prompt: Task-05-03 Trainer ↔ 대시보드 연결

## Purpose
trainer의 학습 루프에서 매 epoch 종료 시 시각화에 필요한 데이터를 수집하여 대시보드 서버로 push하는 연결 로직을 구현한다.

## Context
- 프로젝트는 `bot/plans/plan-dl-cr-dashboard/` 안에서 진행한다.
- Language: Python 3.9+, PyTorch
- Task 06 (Trainer): trainer.py의 callback 시스템 활용
- Task 05-01 (Dashboard Server): DashboardServer.push_update(data)
- WebSocket 메시지 포맷: common_context_prompt.md 참조

## Task Description
1. `dashboard/hook.py`에 DashboardHook 클래스를 구현한다.
   - trainer의 callback 기반 설계와 호환
   - __call__(epoch, train_loss, val_loss, best_val_loss, model, optimizer, trainer) 메서드
   - 매 epoch마다 val dataloader에서 샘플 1개 추출
   - 모델 inference로 prediction 생성
   - 시각화 데이터 수집 및 JSON 직렬화
   - trainer.dashboard_server.push_update(data) 호출

2. 데이터 수집 (JSON 형식)
   ```json
   {
     "epoch": int,
     "total_epochs": int,
     "lr": float,
     "train_loss": float,
     "val_loss": float,
     "best_val": float,
     "train_losses": list,
     "val_losses": list,
     "sample": {
       "input_struct": [[flat 128x128 list]],
       "gt_bggr": [[2x2x30]],
       "pred_bggr": [[2x2x30]],
       "waves": [400.0, 410.3, ...]
     }
   }
   ```

3. 이미지 데이터 처리
   - 128x128 structure: flat list로 변환 (또는 base64 PNG)
   - BGGR 2x2 mean: per-channel mean → (2, 2) array

4. Trainer 수정 (trainer.py)
   - __init__에서 dashboard_server 초기화
   - if config["dashboard"]["enabled"]: server.start()
   - DashboardHook을 callback으로 등록: trainer.add_callback(DashboardHook(trainer))

5. Error handling
   - push 실패 시: try-except로 log하고 계속 진행
   - 대시보드 서버 미실행 시: graceful fallback

6. Wavelength 배열
   - 301 bins 중 out_len(30) 개 선택된 wavelength 계산
   - common_context_prompt.md의 데이터 규약 참조

## Inputs
- `bot/plans/plan-dl-cr-dashboard/common_context_prompt.md` (WebSocket 메시지 규약)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/trainer.py` (Task 06)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/dashboard/server.py` (Task 05-01)

## Outputs
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/dashboard/hook.py` (새로 작성)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/trainer.py` (업데이트)

## Constraints
- DashboardHook은 callable, signature: callback(epoch, train_loss, val_loss, best_val_loss, model, optimizer, trainer)
- 대시보드 서버 미실행 시에도 trainer는 정상 작동
- push_update 호출은 try-except로 보호
- val dataloader에서 첫 번째 배치만 사용
- 이미지 데이터는 numpy array → list 변환 후 JSON 직렬화
- config["dashboard"]["enabled"] == True일 때만 서버 시작

## Implementation Rules
- Python 3.9+ 호환
- PyTorch no_grad() 컨텍스트 사용 (메모리 효율)
- 주석은 한국어 허용, 변수명/함수명은 영어

## Expected Result / Acceptance Criteria
- [ ] DashboardHook 클래스 구현 및 trainer에 등록 가능
- [ ] 매 epoch마다 val sample 1개의 시각화 데이터 수집
- [ ] WebSocket으로 JSON 메시지 전송
- [ ] 브라우저 대시보드에서 실시간 갱신 표시
- [ ] 대시보드 서버 미실행 시에도 trainer 정상 작동
- [ ] push 실패 시 에러 로그하고 계속 진행
- [ ] Wavelength 배열이 정확하게 계산됨
