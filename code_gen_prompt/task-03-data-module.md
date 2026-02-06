# Coding Prompt: Task-03 데이터 로딩 및 전처리 모듈

## Purpose
데이터 로딩, RGB→BGGR 변환, wavelength binning, 180도 augmentation, train/val split을 담당하는 재사용 가능한 데이터 모듈을 구현한다.

## Context
- 프로젝트는 `bot/plans/plan-dl-cr-dashboard/` 안에서 진행한다.
- 데이터 원본: `data_CR-main/` 에 4개 .npy 파일 존재
  - `binary_dataset_128_0.npy`, `binary_dataset_128_1.npy`: (100000,1,128,128) uint8 binary [0,1]
  - `spectra_latest_0.npy`, `spectra_latest_1.npy`: (100000,3,301) float32, 값 범위 [-0.67, 0]
- Language: Python 3.9+, PyTorch
- config의 data 섹션에서 파라미터를 받아야 한다
- 데이터 경로는 config에서 `../data_CR-main/` 상대경로로 지정됨

## Task Description
1. `data/dataset.py`에 `CRDataset(torch.utils.data.Dataset)` 클래스를 구현한다.
   - 여러 .npy 파일을 concat하여 전체 데이터셋 구성
   - numpy mmap_mode="r"로 메모리 효율적 로딩
   - RGB(3,301) → BGGR(2,2,301) 변환: [0,0]=B(ch2), [0,1]=G(ch1), [1,0]=G(ch1), [1,1]=R(ch0)
   - 301 bins → out_len(기본 30) uniform downsample
   - pm1 매핑 옵션: 구조 이미지 [0,1] → [-1,1]
   - 180도 회전 augmentation: 구조 flip(both axes) + BGGR [0,0]↔[1,1] 교환 (B/R swap)
2. `data/dataset.py`에 `create_dataloaders(cfg)` 함수를 구현한다.
   - config dict를 받아 train_loader, val_loader 쌍 반환
   - seed 기반 `torch.utils.data.random_split`으로 재현 가능한 split
3. `data/__init__.py`에서 외부 import 가능하게 export한다.
4. 단독 실행 시(`if __name__ == "__main__"`) shape/sample 출력하는 간단한 테스트를 포함한다.

## Inputs
- `bot/plans/plan-dl-cr-dashboard/common_context_prompt.md` (데이터 규약 참조)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/configs/default.yaml` (data 섹션)
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/data/data_summary.md` (데이터 사양)

## Outputs
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/data/dataset.py`
- `bot/plans/plan-dl-cr-dashboard/code/CR_recon/data/__init__.py` (업데이트)

## Constraints
- numpy mmap_mode="r"로 로딩 (메모리 효율)
- RGB→BGGR 변환 정확성: R=ch0→[1,1], G=ch1→[0,1]&[1,0], B=ch2→[0,0]
- 301→out_len downsample: uniform spacing으로 bin 선택
- 180도 augmentation: 구조 이미지 flip(H,W 모두) + BGGR B/R 교환
- pm1 매핑: `x * 2 - 1`
- train/val split: `torch.Generator` + `manual_seed(cfg["seed"])` 사용
- 코드 내 하드코딩 금지, 모든 파라미터는 config에서 받음

## Implementation Rules
- Python 3.9+ 호환
- PyTorch Dataset/DataLoader 사용
- 주석은 한국어 허용, 변수명/함수명은 영어
- `utils.py`의 `load_config` 사용 가능 (단독 테스트 시)

## Expected Result / Acceptance Criteria
- [ ] CRDataset이 config에서 파라미터를 받아 동작
- [ ] `__getitem__`이 (struct, spectrum) 튜플 반환: struct=(1,128,128) float32, spectrum=(2,2,out_len) float32
- [ ] create_dataloaders(cfg)가 train_loader, val_loader 반환
- [ ] augmentation on/off가 config로 제어 가능
- [ ] 여러 .npy 파일 concat 지원
- [ ] 단독 실행 시 shape/sample 출력
