# Task 03: 데이터 로딩 및 전처리 모듈

## 목적
데이터 로딩, BGGR 변환, wavelength binning, 180도 augmentation, train/val split을 담당하는 재사용 가능한 데이터 모듈을 구현한다.

## 입력
- `CR_recon/data/` 내 .npy 파일들
- `configs/default.yaml`의 data 섹션 (bin 수, augmentation 옵션, split 비율, batch_size 등)

## 출력
- `CR_recon/data/dataset.py` — Dataset 클래스 + DataLoader 생성 유틸
- `CR_recon/data/__init__.py` — 외부 import용

## 제약
- numpy mmap_mode="r"로 메모리 효율적 로딩
- RGB→BGGR(2x2) 변환: R→[1,1], G→[0,1]&[1,0], B→[0,0]
- 301 bins → configurable out_len(기본 30) downsample
- 180도 회전 augmentation: 구조 flip + B/R 스펙트럼 교환
- pm1 매핑 (0,1 → -1,1) 옵션
- 여러 .npy 파일 concat 지원 (0번 + 1번 파일 합치기)
- train/val split은 seed 고정 재현 가능

## 완료 조건
- [ ] Dataset 클래스가 config에서 파라미터를 받아 동작
- [ ] DataLoader 생성 함수가 train/val loader 쌍을 반환
- [ ] augmentation on/off가 config로 제어 가능
- [ ] 단독 실행 시 shape/sample 출력하는 간단한 테스트 포함
