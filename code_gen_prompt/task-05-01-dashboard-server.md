# Coding Prompt: Task-05-01 대시보드 백엔드 서버

## Purpose
학습 상태 데이터를 WebSocket으로 브라우저에 실시간 전송하는 경량 서버를 구현한다.

## Context
- 프로젝트는 `bot/plans/plan-dl-cr-dashboard/` 안에서 진행한다.
- Language: Python 3.9+, FastAPI, uvicorn
- 서버 포트: config["dashboard"]["port"] (기본 8501)
- WebSocket 메시지 포맷: common_context_prompt.md의 WebSocket 메시지 규약 참조
- 학습 루프와 분리: 별도 스레드에서 구동 (trainer callback으로 데이터 push)
- 서버 라이프사이클: start() / stop() / is_running() 메서드 필수

## Task Description
1. `dashboard/server.py`에 DashboardServer 클래스를 구현한다.
   - FastAPI 앱 구성
   - WebSocket endpoint: `/ws` — 학습 데이터 실시간 전송
   - REST endpoint: `/api/status` — 현재 학습 상태 JSON 반환
   - Static 파일 서빙: `/` → `dashboard/static/index.html`
   - 별도 스레드에서 uvicorn 서버 구동 (threading.Thread)

2. 상태 관리
   - in-memory dict: 최신 학습 상태 (epoch, train_loss, val_loss, best_val, lr, sample)
   - WebSocket 클라이언트 관리: 메시지 broadcast

3. 메서드
   - `start(port)`: 서버 시작 (별도 스레드)
   - `stop()`: 서버 종료
   - `is_running()`: 서버 실행 상태
   - `push_update(data: dict)`: 학습 데이터 push (trainer callback에서 호출)

4. WebSocket 메시지 포맷 (common_context_prompt.md 참조)
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

5. Error handling
   - WebSocket 연결 실패 시 graceful handling
   - 서버 비활성화 시 push_update 무시

6. `dashboard/__init__.py` 업데이트
   - DashboardServer export

## Inputs
- `bot/plans/plan-dl-cr-dashboard/common_context_prompt.md` (WebSocket 메시지 규약)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/configs/default.yaml`

## Outputs
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/dashboard/server.py`
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/dashboard/__init__.py` (업데이트)

## Constraints
- FastAPI + uvicorn 사용 (빠르고 가볍고 async 지원)
- 별도 스레드에서 구동 (trainer의 메인 루프 blocking 금지)
- WebSocket endpoint: `/ws`
- REST endpoint: `/api/status`
- Static 파일 경로: 상대경로 기반 (dashboard/static/)
- config["dashboard"]["enabled"] == True일 때만 시작
- WebSocket push 실패 시 log만 하고 계속 진행 (예외 발생 금지)

## Implementation Rules
- Python 3.9+ 호환
- FastAPI, uvicorn 사용
- threading.Thread로 별도 스레드 구동
- 주석은 한국어 허용, 변수명/함수명은 영어
- asyncio + WebSocket 호환

## Expected Result / Acceptance Criteria
- [ ] DashboardServer 클래스 구현
- [ ] `start()` 메서드로 별도 스레드에서 서버 시작 가능
- [ ] `stop()` 메서드로 안전하게 종료 가능
- [ ] `is_running()` 메서드로 상태 확인 가능
- [ ] `push_update(data)` 메서드로 WebSocket 클라이언트에 데이터 전송
- [ ] `/api/status` endpoint에서 JSON 반환
- [ ] `/` endpoint에서 index.html 서빙
- [ ] WebSocket 메시지 포맷이 common_context_prompt.md와 일치
- [ ] 서버 비활성화 상태에서도 trainer 정상 작동
