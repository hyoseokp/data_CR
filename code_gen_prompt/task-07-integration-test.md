# Coding Prompt: Task-07 통합 테스트 및 검증

## Purpose
전체 파이프라인(데이터→모델→loss→trainer→대시보드)이 config 하나로 올바르게 동작하는지 검증하고, 모델/loss 교체가 실제로 한 줄 변경으로 가능한지 확인한다.

## Context
- 프로젝트는 `bot/plans/plan-dl-cr-dashboard/code/CR_recon/` 안에서 실행된다.
- Task 01~06의 모든 산출물 완성됨
- 기본 config: configs/default.yaml (MetaSpec_CNNXAttn + MSE+Pearson loss)

## Task Description
1. `configs/test_cnn_gru.yaml` 작성
   - 기본 설정은 default.yaml과 동일
   - model.name: "cnn_gru"
   - model.params: d_model=128, gru_layers=2 (간단한 설정)
   - loss.name: "weighted_smooth"
   - loss.params: lambda1=0.1, lambda2=0.05
   - training.epochs: 5 (테스트용 짧은 학습)

2. 통합 테스트 시나리오 5가지
   - **시나리오 1**: 기본 실행 (MetaSpec_CNNXAttn + MSE+Pearson, 5 epoch, 대시보드 활성)
     - 명령: `cd code/CR_recon && python train.py --config configs/default.yaml`
     - 검증: 학습 정상 완료, 체크포인트 저장, 로그 파일 생성
     - 대시보드: http://localhost:8501 접속 후 6개 패널 실시간 갱신 확인

   - **시나리오 2**: 모델 교체 (MetaSpec_CNNGRU + MSE+Pearson, 5 epoch)
     - 명령: `python train.py --config configs/test_cnn_gru.yaml --resume outputs/cnn_xattn_best.pt` (선택사항)
     - 검증: CNNGRU 모델 로드, 학습 정상 완료

   - **시나리오 3**: Loss 교체 (MetaSpec_CNNXAttn + Weighted+Smooth, 5 epoch)
     - config 수정: loss.name: "weighted_smooth", training.epochs: 5
     - 명령: `python train.py --config configs/default_weighted.yaml`
     - 검증: Weighted loss 적용, 학습 정상 완료

   - **시나리오 4**: 대시보드 비활성화 (MetaSpec_CNNXAttn, 5 epoch, 대시보드 OFF)
     - config 수정: dashboard.enabled: false
     - 명령: `python train.py --config configs/default_no_dashboard.yaml`
     - 검증: 대시보드 서버 미시작, 학습 정상 완료 (에러 없음)

   - **시나리오 5**: Checkpoint Resume (MetaSpec_CNNXAttn, 이어서 학습)
     - 명령: `python train.py --config configs/default.yaml --resume outputs/cnn_xattn_best.pt`
     - 검증: 이전 체크포인트에서 학습 재개, epoch 번호 계속됨

3. 테스트 결과 문서화
   - 각 시나리오별 성공/실패 기록
   - 대시보드 스크린샷 또는 텍스트 로그
   - 최종 통합 테스트 결과 보고서

## Inputs
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/configs/default.yaml`
- Task 01~06의 모든 코드 파일

## Outputs
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/configs/test_cnn_gru.yaml`
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/configs/default_weighted.yaml` (선택사항)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/configs/default_no_dashboard.yaml` (선택사항)
- 통합 테스트 결과 로그 (텍스트 문서)

## Constraints
- 테스트 실행은 선택사항 (코드 작성만 요구 시 config 파일 작성으로 충분)
- 각 시나리오 실행 가능 (권장)
- requirements.txt에 모든 의존성 명시
- 테스트용 config는 training.epochs=5 이상 (빠른 검증용)

## Implementation Rules
- YAML config 파일 작성
- Python은 새로 작성하지 않음 (기존 코드 재사용)
- 각 config 파일은 주석으로 용도 명시

## Expected Result / Acceptance Criteria
- [ ] test_cnn_gru.yaml, default_weighted.yaml, default_no_dashboard.yaml 작성
- [ ] 각 config 파일이 valid YAML (오류 없음)
- [ ] 시나리오 1: 기본 실행 성공 (대시보드 6개 패널 갱신 확인)
- [ ] 시나리오 2: 모델 교체 성공 (MetaSpec_CNNGRU 모델 로드)
- [ ] 시나리오 3: Loss 교체 성공 (Weighted loss 적용)
- [ ] 시나리오 4: 대시보드 비활성화 시 학습 정상 완료
- [ ] 시나리오 5: Checkpoint resume 정상 작동
- [ ] requirements.txt에 모든 의존성 명시
