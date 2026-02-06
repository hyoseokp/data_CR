# Coding Prompt: Task-06 학습 엔진 (Trainer)

## Purpose
config 기반으로 모델/loss/데이터를 조립하여 학습을 실행하는 trainer를 구현한다.

## Context
- 프로젝트는 `bot/plans/plan-dl-cr-dashboard/` 안에서 진행한다.
- Language: Python 3.9+, PyTorch
- Task 03 (데이터 모듈): CRDataset, create_dataloaders
- Task 04 (모델/loss 레지스트리): get_model, get_loss
- config: configs/default.yaml에서 모든 파라미터 제어
- 기본 설정: epochs=300, lr=1e-3, weight_decay=5e-3, grad_clip=1.0, use_amp=true, warmup_ratio=0.05, save_every=10

## Task Description
1. `trainer.py`에 Trainer 클래스를 구현한다.
   - config dict를 받아 초기화
   - model, optimizer, scheduler, dataloaders를 조립
   - AMP (mixed precision) 지원 (torch.amp.autocast, GradScaler)
   - Gradient clipping (설정 가능한 max_norm)
   - Cosine annealing with linear warmup scheduler
   - Callback 리스트 주입 가능 (대시보드 hook 등)
   - train_one_epoch, validate, save_checkpoint 메서드

2. `train.py`에 CLI 진입점을 구현한다.
   - argparse로 --config 옵션
   - config 로드 → Trainer 생성 → train() 호출
   - 단순하고 간결한 구조

3. Checkpoint 저장 구조
   - best: val loss 기준 최고 성능
   - last: 마지막 epoch
   - periodic: N epoch마다 (config의 save_every)
   - 각 checkpoint에 포함: model state_dict, optimizer state_dict, epoch, val_loss, best_val

4. 로깅
   - 로그 파일: config의 output.log_file 경로에 기록
   - epoch별: train_loss, val_loss, best_val, lr, epoch 정보

5. Progress bar
   - tqdm으로 train/val loop 진행 상황 표시

## Inputs
- `bot/plans/plan-dl-cr-dashboard/common_context_prompt.md` (Config Schema, 학습 파라미터)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/configs/default.yaml`
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/data/dataset.py` (create_dataloaders)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/models/__init__.py` (get_model)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/losses/__init__.py` (get_loss)

## Outputs
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/trainer.py`
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/train.py`

## Constraints
- AMP on/off: config["training"]["use_amp"]에서 제어
- Gradient clipping: config["training"]["grad_clip"]
- Scheduler: cosine annealing + linear warmup (config["training"]["warmup_ratio"])
- Callback 기반 확장성 (trainer.add_callback(hook))
- 체크포인트는 output 디렉토리에 저장 (config["output"]["dir"])
- 로그 파일은 단순 텍스트 (.txt) 형식
- 대시보드 서버 미실행 시에도 학습은 정상 진행 (graceful fallback)

## Implementation Rules
- Python 3.9+ 호환
- PyTorch nn.Module, Optimizer, Scheduler 사용
- 주석은 한국어 허용, 변수명/함수명은 영어
- Callback signature: callback(epoch, train_loss, val_loss, model, optimizer, ...)
- train() 메서드는 전체 학습 루프 실행

## Expected Result / Acceptance Criteria
- [ ] `python train.py --config configs/default.yaml` 실행 시 학습 시작
- [ ] AMP on/off config 제어 동작
- [ ] Gradient clipping 동작
- [ ] Cosine annealing + warmup scheduler 동작
- [ ] best/last/periodic checkpoint 저장
- [ ] 로그 파일에 epoch별 정보 기록
- [ ] callback 리스트 주입 가능
- [ ] checkpoint resume 지원 (--resume 옵션 선택사항)
