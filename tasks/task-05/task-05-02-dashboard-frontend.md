# Task 05-02: 대시보드 프론트엔드

## 목적
브라우저에서 학습 과정을 실시간으로 시각화하는 단일 HTML 페이지를 구현한다.

## 입력
- Task 05-01의 WebSocket 메시지 사양 (JSON 구조)

## 출력
- `CR_recon/dashboard/static/index.html` — 단일 파일 (HTML + CSS + JS 인라인)

## 제약
- 외부 CDN 사용 가능: Chart.js (차트), 필요 시 Plotly.js
- 빌드 도구 없이 단일 HTML 파일로 동작
- WebSocket 자동 재연결 (연결 끊김 시 retry)

## 시각화 패널 구성 (6개)
1. **Loss 곡선** — train/val loss, log scale, epoch별 추이
2. **GT vs Prediction 스펙트럼** — B(파랑), G(초록), R(빨강) 각각 GT(점선) vs Pred(실선), wavelength(nm) x축
3. **Input Structure** — 128x128 grayscale 이미지 (canvas)
4. **GT BGGR 2x2 mean** — 2x2 격자를 확대한 grayscale 이미지
5. **Pred BGGR 2x2 mean** — 2x2 격자를 확대한 grayscale 이미지
6. **R/G/B Abs Error** — 채널별 절대 오차 곡선, wavelength(nm) x축

## 완료 조건
- [ ] 6개 패널이 2x3 또는 3x2 grid로 배치
- [ ] WebSocket 메시지 수신 시 모든 차트 자동 갱신
- [ ] 연결 상태 표시 (connected/disconnected)
- [ ] 현재 epoch, learning rate, best val loss 텍스트 표시
