# Task 02: 프로젝트 구조 설계 및 Config 시스템

## 목적
모듈화된 프로젝트 디렉토리 구조를 생성하고, YAML config 파일 하나로 모델/loss/하이퍼파라미터를 교체할 수 있는 설정 시스템을 만든다.

## 입력
- Task 01에서 확인된 데이터 사양 (data_summary.md)

## 출력
- 프로젝트 루트: `CR_recon/`
- 디렉토리 구조:
  ```
  CR_recon/
  ├── configs/         # YAML config 파일들
  │   └── default.yaml
  ├── data/            # 데이터 파일 + 데이터 모듈
  │   └── dataset.py
  ├── models/          # 모델 레지스트리 + 모델 정의
  │   └── __init__.py
  ├── losses/          # Loss 레지스트리 + loss 정의
  │   └── __init__.py
  ├── dashboard/       # 실시간 대시보드 서버 + 프론트엔드
  ├── trainer.py       # 학습 엔진
  ├── train.py         # 진입점 (config 로드 → 학습 실행)
  └── requirements.txt
  ```
- `configs/default.yaml` (모델명, loss명, lr, epochs, batch_size 등 포함)

## 제약
- config는 YAML 단일 파일로 관리
- 모델/loss 교체는 config의 `model.name`, `loss.name` 필드 변경만으로 가능해야 함
- 기존 Research_CR_NA .ipynb의 하이퍼파라미터를 default로 반영

## 완료 조건
- [ ] 디렉토리 구조 생성 완료
- [ ] default.yaml에 모든 주요 하이퍼파라미터 포함
- [ ] config 로딩 유틸리티 함수 사양 정의
