# Coding Prompt: Task-05-02 대시보드 프론트엔드

## Purpose
브라우저에서 학습 과정을 실시간으로 시각화하는 단일 HTML 페이지를 구현한다.

## Context
- WebSocket으로 server.py로부터 실시간 데이터 수신
- 외부 CDN 사용 가능: Chart.js (차트 그리기)
- 빌드 도구 없이 단일 HTML 파일로 동작
- 포트: 기본 http://localhost:8501

## Task Description
1. `dashboard/static/index.html` 작성: 단일 HTML 파일 (HTML + CSS + JS 인라인)
   - 내부 <style> 태그에 CSS
   - 내부 <script> 태그에 JavaScript
   - 외부 라이브러리: Chart.js (CDN)

2. 시각화 패널 (6개)
   - **Panel 1**: Loss 곡선 (train/val, log scale)
   - **Panel 2**: GT vs Prediction 스펙트럼 (B/G/R 각각 GT 점선 vs Pred 실선)
   - **Panel 3**: Input Structure (128x128 grayscale canvas)
   - **Panel 4**: GT BGGR 2x2 mean (2x2 격자 확대 이미지)
   - **Panel 5**: Pred BGGR 2x2 mean (2x2 격자 확대 이미지)
   - **Panel 6**: R/G/B Abs Error (채널별 오차 곡선)

3. 레이아웃
   - 상단: 헤더 (제목, 연결 상태, 현재 epoch, learning rate, best val loss)
   - 중앙: 2x3 grid 또는 3x2 grid로 6개 패널 배치
   - 반응형 웹디자인 권장

4. WebSocket 연결
   - 자동 연결: ws://localhost:8501/ws
   - 수신 메시지: JSON 형식 (common_context_prompt.md의 WebSocket 규약)
   - 자동 재연결: 연결 끊김 시 3초마다 재시도

5. 상태 표시
   - 연결 상태: "Connected" (초록색) / "Disconnected" (빨강색)
   - 현재 epoch / total_epochs
   - learning rate (scientific notation)
   - best_val (best validation loss)

6. 차트 갱신
   - 메시지 수신 시 모든 차트 자동 갱신
   - Chart.js 라이브러리 사용

## Inputs
- `bot/plans/plan-dl-cr-dashboard/common_context_prompt.md` (WebSocket 메시지 규약, 패널 사양)

## Outputs
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/dashboard/static/index.html`

## Constraints
- 단일 HTML 파일 (빌드 도구 없음)
- 외부 CDN 사용: Chart.js (https://cdn.jsdelivr.net/npm/chart.js)
- 내부 CSS/JS 인라인
- 포트 8501 기본값
- WebSocket URL: ws://localhost:8501/ws (런타임에 동적 설정 가능)
- 자동 재연결: exponential backoff (3초, 6초, 12초, ... 최대 60초)

## Implementation Rules
- 순수 HTML + CSS + JavaScript (프레임워크 없음)
- Chart.js CDN 사용
- 모던 브라우저 호환 (ES6+ 가능)
- 주석은 한국어 허용

## Expected Result / Acceptance Criteria
- [ ] index.html 단일 파일 생성
- [ ] 6개 패널이 2x3 grid로 배치
- [ ] WebSocket 자동 연결
- [ ] 메시지 수신 시 차트 자동 갱신
- [ ] 연결 상태 표시 (connected/disconnected)
- [ ] 현재 epoch, lr, best_val 표시
- [ ] 차트 범례 명확 (train/val, GT/Pred, B/G/R 등)
- [ ] 모바일 반응형 레이아웃 (선택사항, 권장)
