# Task 05: 실시간 대시보드 (서버 + 프론트엔드 + trainer 연결)

## 목적
학습 중 매 epoch마다 브라우저에서 실시간으로 학습 상태를 시각화하는 대시보드를 구현한다.
서버, 프론트엔드, trainer 연결을 포함하는 복합 task이므로 subtask로 분해한다.

## 전체 구성
1. **task-05-01**: 대시보드 백엔드 서버 (WebSocket + REST)
2. **task-05-02**: 대시보드 프론트엔드 (HTML + JS 차트)
3. **task-05-03**: trainer → 대시보드 데이터 push 연결

## 최종 출력
- `CR_recon/dashboard/server.py`
- `CR_recon/dashboard/static/index.html`
- `CR_recon/trainer.py` (대시보드 hook 포함)
