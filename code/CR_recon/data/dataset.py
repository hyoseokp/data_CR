"""
데이터 로딩 및 전처리 모듈
RGB→BGGR 변환, wavelength binning, 180도 augmentation, train/val split
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


class CRDataset(Dataset):
    """
    128x128 구조 이미지 → Bayer 2x2 스펙트럼 (301 bins) 예측 데이터셋

    데이터 정제 파이프라인:
    1. 원본: struct (N, 1, 128, 128), spectra (N, 3, 301)
    2. 필터링: 유효한 샘플만 선택 (spectra ≠ 0)
    3. Bayer 변환: spectra (M, 3, 301) → (M, 2, 2, out_len)
       - [0,0]=R, [0,1]=G, [1,0]=G, [1,1]=B
    4. 180도 증강: Bayer의 R과 B 위치 교환 (augment=True일 때)
    """

    def __init__(self, cfg, augment=False):
        """
        cfg: dict with keys:
          - struct_files: list of struct .npy paths (relative to config dir)
          - spectra_files: list of spectra .npy paths
          - out_len: output wavelength bins (default 30)
          - map_to_pm1: bool, map struct [0,1] → [-1,1]
          - augment_180: bool, enable 180° rotation data duplication
        augment: bool, whether to include 180° rotated samples (train=True, val=False)
        """
        self.cfg = cfg
        self.augment = augment and cfg.get("augment_180", False)
        self.out_len = cfg.get("out_len", 30)
        self.map_to_pm1 = cfg.get("map_to_pm1", True)

        # 데이터 경로
        cfg_dir = Path(__file__).parent.parent  # CR_recon/

        # 정제된 데이터 경로 (사전에 전처리된 Bayer 파일)
        # 위치: CR_recon/dataset/bayer/
        bayer_dir = cfg_dir / "dataset" / "bayer"
        struct_bayer_paths = [
            bayer_dir / "struct_0.npy",
            bayer_dir / "struct_1.npy"
        ]
        bayer_paths = [
            bayer_dir / "bayer_0.npy",
            bayer_dir / "bayer_1.npy"
        ]
        bayer_rotated_paths = [
            bayer_dir / "bayer_rotated_0.npy",
            bayer_dir / "bayer_rotated_1.npy"
        ]

        # 정제된 데이터 있는지 확인
        if all(p.exists() for p in struct_bayer_paths + bayer_paths + bayer_rotated_paths):
            # 정제된 데이터 로드 (이미 필터링되고 변환됨)
            structs_valid = [np.load(str(p), mmap_mode="r") for p in struct_bayer_paths]
            spectra_valid = [np.load(str(p), mmap_mode="r") for p in bayer_paths]
            spectra_rotated_list = [np.load(str(p), mmap_mode="r") for p in bayer_rotated_paths]

            struct_valid = np.concatenate(structs_valid, axis=0)  # (M, 1, 128, 128)
            spectra_valid = np.concatenate(spectra_valid, axis=0)  # (M, 2, 2, out_len)
            bayer_rotated_valid = np.concatenate(spectra_rotated_list, axis=0)  # (M, 2, 2, out_len)
        else:
            # 정제된 데이터 없으면 원본에서 즉시 정제
            struct_paths = [cfg_dir / p for p in cfg["struct_files"]]
            spectra_paths = [cfg_dir / p for p in cfg["spectra_files"]]

            # numpy mmap으로 로드 (메모리 효율적)
            structs = [np.load(str(p), mmap_mode="r") for p in struct_paths]
            spectra = [np.load(str(p), mmap_mode="r") for p in spectra_paths]

            # concat (여러 파일 합치기)
            struct_all = np.concatenate(structs, axis=0)  # (N, 1, 128, 128)
            spectra_all = np.concatenate(spectra, axis=0)  # (N, 3, 301)

            # 데이터 정제: 0이 아닌 유효한 샘플만 필터링
            valid_indices = np.where(np.any(spectra_all != 0, axis=(1, 2)))[0]
            struct_valid = struct_all[valid_indices]  # (M, 1, 128, 128)
            spectra_valid = spectra_all[valid_indices]  # (M, 3, 301)

            # spectra를 (3, 301) → (2, 2, out_len) Bayer 패턴으로 변환
            spectra_valid = self._convert_to_bayer(spectra_valid)  # (M, 2, 2, out_len)

            # 180도 회전된 버전 생성
            bayer_rotated_valid = np.zeros_like(spectra_valid)
            bayer_rotated_valid[:, 0, 0, :] = spectra_valid[:, 1, 1, :]
            bayer_rotated_valid[:, 0, 1, :] = spectra_valid[:, 0, 1, :]
            bayer_rotated_valid[:, 1, 0, :] = spectra_valid[:, 1, 0, :]
            bayer_rotated_valid[:, 1, 1, :] = spectra_valid[:, 0, 0, :]

        # 180도 회전된 구조 이미지 생성 (대각선 대칭 구조 활용)
        struct_rotated = np.flip(struct_valid, axis=(2, 3)).copy()  # (M, 1, 128, 128)

        # 원본 + 180도 회전 버전 병합
        if self.augment:
            self.struct = np.concatenate([struct_valid, struct_rotated], axis=0)  # (2M, 1, 128, 128)
            self.spectra = np.concatenate([spectra_valid, bayer_rotated_valid], axis=0)  # (2M, 2, 2, out_len)
        else:
            self.struct = struct_valid
            self.spectra = spectra_valid  # (M, 2, 2, out_len)

        self.n = len(self.struct)

    def _convert_to_bayer(self, spectra):
        """
        (M, 3, 301) → (M, 2, 2, out_len) Bayer 패턴 변환

        입력: spectra (M, 3, 301) = [R, G, B] × 301파장
        출력: bayer (M, 2, 2, out_len) = [[R, G], [G, B]] × out_len파장
        """
        M = len(spectra)
        bayer = np.zeros((M, 2, 2, self.out_len), dtype=np.float32)

        # 다운샘플 indices
        indices = np.linspace(0, 300, self.out_len, dtype=int)

        # 배치 처리
        for i in range(M):
            spec = spectra[i].astype(np.float32)  # (3, 301)
            r, g, b = spec[0], spec[1], spec[2]

            # 먼저 (2, 2, 301) Bayer 패턴 구성
            bggr_full = np.zeros((2, 2, 301), dtype=np.float32)
            bggr_full[0, 0, :] = r  # [0, 0] = R
            bggr_full[0, 1, :] = g  # [0, 1] = G
            bggr_full[1, 0, :] = g  # [1, 0] = G
            bggr_full[1, 1, :] = b  # [1, 1] = B

            # 다운샘플 (301 → out_len)
            bayer[i] = bggr_full[:, :, indices]

        return bayer

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        """
        Returns: (struct, spectrum)
          - struct: (1, 128, 128) float32
          - spectrum: (2, 2, out_len) float32 Bayer pattern

        augment_180이 활성화된 경우:
          - idx 0~M-1: 원본 데이터
          - idx M~2M-1: 180도 회전된 데이터
        (Bayer 패턴 변환은 이미 __init에서 수행됨)
        """
        # 구조 이미지: (1, 128, 128) uint8 → float32
        struct = self.struct[idx].astype(np.float32)  # (1, 128, 128)

        if self.map_to_pm1:
            struct = struct * 2.0 - 1.0  # [0,1] → [-1,1]

        # 스펙트럼: 이미 (2, 2, out_len) Bayer 패턴으로 변환됨
        bggr = self.spectra[idx].astype(np.float32)  # (2, 2, out_len)

        # 부호 변경: 음수 데이터를 양수로 변환
        bggr = -bggr

        return torch.from_numpy(struct), torch.from_numpy(bggr)


def create_dataloaders(cfg):
    """
    config dict로부터 train/val DataLoader 쌍 생성

    augment_180이 활성화되면:
      - Train set: 원본 N + 180도 회전 N = 2N개 샘플
      - Val set: 원본만 M개 샘플

    Returns: (train_loader, val_loader)
    """
    # seed 고정 (재현성)
    seed = cfg.get("seed", 42)
    g = torch.Generator()
    g.manual_seed(seed)

    # 데이터셋 생성 (augment=True로 train, augment=False로 val)
    train_dataset = CRDataset(cfg["data"], augment=True)
    val_dataset = CRDataset(cfg["data"], augment=False)

    # train/val split (val_dataset은 원본만 있음)
    train_ratio = cfg["data"].get("train_ratio", 0.95)
    train_size = int(len(val_dataset) * train_ratio)
    val_size = len(val_dataset) - train_size

    # val_dataset의 indices로 split
    _, val_subset = random_split(
        val_dataset, [train_size, val_size], generator=g
    )

    # train_dataset에서도 동일한 indices의 샘플들만 사용
    train_indices = list(range(train_size))
    # augment_180이 활성화되면, 원본 0~train_size-1 + 회전 train_size~2*train_size-1
    if cfg["data"].get("augment_180", False):
        # 원본 인덱스들: 0 ~ train_size-1
        # 회전 인덱스들: train_size ~ 2*train_size-1
        train_indices = list(range(train_size)) + list(range(train_size, 2 * train_size))

    class SubsetDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

    train_subset = SubsetDataset(train_dataset, train_indices)

    # DataLoader 생성
    batch_size = cfg["data"].get("batch_size", 64)
    num_workers = cfg["data"].get("num_workers", 0)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


if __name__ == "__main__":
    """
    단독 실행 테스트: shape 및 sample 출력
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from utils import load_config

    # config 로드
    cfg = load_config("../configs/default.yaml")
    print(f"Config loaded: {cfg}")

    # 데이터셋 생성 (augment 없음)
    print("\n데이터 로딩 중...")
    dataset = CRDataset(cfg["data"], augment=False)
    print(f"✓ Dataset size (after filtering valid samples): {len(dataset)}")

    # 샘플 출력
    struct, spec = dataset[0]
    print(f"\n첫 번째 샘플:")
    print(f"  struct shape: {struct.shape}, dtype: {struct.dtype}")
    print(f"  spectrum shape: {spec.shape}, dtype: {spec.dtype}")
    print(f"  struct range: [{struct.min():.4f}, {struct.max():.4f}]")
    print(f"  spectrum range: [{spec.min():.4f}, {spec.max():.4f}]")

    # DataLoader 생성
    print("\nDataLoader 생성 중...")
    train_loader, val_loader = create_dataloaders(cfg)
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")

    # 첫 배치 확인
    batch_struct, batch_spec = next(iter(train_loader))
    print(f"\n첫 배치:")
    print(f"  struct shape: {batch_struct.shape}")
    print(f"  spectrum shape: {batch_spec.shape}")
