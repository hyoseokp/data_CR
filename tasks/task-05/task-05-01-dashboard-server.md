# Task 05-01: 대시보드 백엔드 서버

## 목적
학습 상태 데이터를 WebSocket으로 브라우저에 실시간 전송하는 경량 서버를 구현한다.

## 입력
- 없음 (trainer에서 호출될 API 사양만 정의)

## 출력
- `CR_recon/dashboard/__init__.py`
- `CR_recon/dashboard/server.py` — FastAPI + WebSocket 서버

## 제약
- FastAPI + uvicorn 사용 (설치 간편, async 지원)
- 별도 스레드에서 서버 구동 (학습 메인 루프 blocking 방지)
- WebSocket endpoint: `/ws` — JSON 메시지로 학습 데이터 push
- REST endpoint: `/api/status` — 현재 학습 상태 조회
- static 파일 서빙: `/` → `dashboard/static/index.html`
- 포트: config에서 설정 가능 (기본 8501)

## 완료 조건
- [ ] 서버가 별도 스레드에서 시작/종료 가능
- [ ] WebSocket 연결 시 클라이언트에 epoch 데이터 전송 가능
- [ ] static 파일 서빙 동작 확인
