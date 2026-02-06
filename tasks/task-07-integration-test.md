# Task 07: 통합 테스트 및 검증

## 목적
전체 파이프라인(데이터→모델→loss→trainer→대시보드)이 config 하나로 올바르게 동작하는지 검증하고, 모델/loss 교체가 실제로 한 줄 변경으로 가능한지 확인한다.

## 입력
- Task 01~06의 모든 산출물
- `configs/default.yaml`

## 출력
- `configs/test_cnn_gru.yaml` — MetaSpec_CNNGRU + weighted_smooth loss 테스트용 config
- 통합 테스트 결과 로그

## 검증 시나리오
1. **기본 실행**: `python train.py --config configs/default.yaml`로 5 epoch 학습, 대시보드 브라우저에서 6개 패널 모두 갱신 확인
2. **모델 교체**: config에서 `model.name: cnn_gru`로 변경 후 학습 정상 동작 확인
3. **Loss 교체**: config에서 `loss.name: weighted_smooth`로 변경 후 학습 정상 동작 확인
4. **대시보드 없이 실행**: 대시보드 서버 비활성화 상태에서 학습 정상 완료 확인
5. **Checkpoint resume**: 저장된 checkpoint에서 학습 재개 가능 확인

## 완료 조건
- [ ] 5개 시나리오 모두 통과
- [ ] config 변경만으로 모델/loss 교체 동작 확인
- [ ] 대시보드 6개 패널 실시간 갱신 스크린샷 또는 로그
- [ ] requirements.txt에 모든 의존성 기재
