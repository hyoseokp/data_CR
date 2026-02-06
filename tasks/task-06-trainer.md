# Task 06: 학습 엔진 (Trainer)

## 목적
config 기반으로 모델/loss/데이터를 조립하여 학습을 실행하는 trainer를 구현한다.

## 입력
- `configs/default.yaml`
- Task 03의 데이터 모듈
- Task 04의 모델/loss 레지스트리

## 출력
- `CR_recon/trainer.py` — Trainer 클래스
- `CR_recon/train.py` — CLI 진입점 (`python train.py --config configs/default.yaml`)

## 제약
- AMP (mixed precision) on/off config 제어
- Gradient clipping (configurable max_norm)
- Cosine annealing with warmup scheduler
- Checkpoint 저장: best (val loss 기준), last, periodic (N epoch마다)
- 체크포인트에 model state, optimizer state, epoch, best_val 포함
- tqdm progress bar
- 로그 파일 출력 (.txt)
- 대시보드 hook 주입 가능한 구조 (callback 리스트)

## 완료 조건
- [ ] `python train.py --config configs/default.yaml` 실행 시 학습 시작
- [ ] AMP, grad_clip, scheduler가 config에서 제어
- [ ] best/last/periodic checkpoint 저장
- [ ] callback 리스트에 대시보드 hook 추가 가능
- [ ] 로그 파일에 epoch별 train/val loss 기록
