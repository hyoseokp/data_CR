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
    128x128 구조 이미지 → BGGR 2x2 스펙트럼 예측 데이터셋
    """

    def __init__(self, cfg, augment=False):
        """
        cfg: dict with keys:
          - struct_files: list of struct .npy paths (relative to config dir)
          - spectra_files: list of spectra .npy paths
          - out_len: output wavelength bins (default 30)
          - map_to_pm1: bool, map struct [0,1] → [-1,1]
          - augment_180: bool, enable 180° rotation augmentation
        augment: bool, whether to apply augmentation (train=True, val=False)
        """
        self.cfg = cfg
        self.augment = augment and cfg.get("augment_180", False)
        self.out_len = cfg.get("out_len", 30)
        self.map_to_pm1 = cfg.get("map_to_pm1", True)

        # 데이터 경로 (config 파일 기준 상대경로)
        cfg_dir = Path(__file__).parent.parent.parent  # CR_recon/
        struct_paths = [cfg_dir / p for p in cfg["struct_files"]]
        spectra_paths = [cfg_dir / p for p in cfg["spectra_files"]]

        # numpy mmap으로 로드 (메모리 효율적)
        structs = [np.load(str(p), mmap_mode="r") for p in struct_paths]
        spectra = [np.load(str(p), mmap_mode="r") for p in spectra_paths]

        # concat (여러 파일 합치기)
        self.struct = np.concatenate(structs, axis=0)  # (N, 1, 128, 128)
        self.spectra = np.concatenate(spectra, axis=0)  # (N, 3, 301)

        # 데이터 정제: 0이 아닌 유효한 샘플만 필터링
        # spectra에서 모든 값이 0인 샘플을 제외
        valid_indices = np.where(np.any(self.spectra != 0, axis=(1, 2)))[0]
        self.struct = self.struct[valid_indices]
        self.spectra = self.spectra[valid_indices]

        self.n = len(self.struct)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        """
        Returns: (struct, spectrum)
          - struct: (1, 128, 128) float32
          - spectrum: (2, 2, out_len) float32 BGGR
        """
        # 구조 이미지: (1, 128, 128) uint8 → float32
        struct = self.struct[idx].astype(np.float32)  # (1, 128, 128)

        if self.map_to_pm1:
            struct = struct * 2.0 - 1.0  # [0,1] → [-1,1]

        # 스펙트럼: (3, 301) float32
        spec = self.spectra[idx].astype(np.float32)  # (3, 301)

        # 부호 변경: 음수 데이터를 양수로 변환
        spec = -spec

        # RGB → BGGR 변환 및 2x2 형태로 reshape
        # spec: [R, G, B] (ch=3, L=301)
        # BGGR: [0,0]=B, [0,1]=G, [1,0]=G, [1,1]=R
        r, g, b = spec[0], spec[1], spec[2]

        # Downsample 301 → out_len (uniform spacing)
        indices = np.linspace(0, 300, self.out_len, dtype=int)
        r = r[indices]
        g = g[indices]
        b = b[indices]

        # BGGR (2, 2, out_len) 구성
        bggr = np.zeros((2, 2, self.out_len), dtype=np.float32)
        bggr[0, 0, :] = b  # [0, 0] = B
        bggr[0, 1, :] = g  # [0, 1] = G
        bggr[1, 0, :] = g  # [1, 0] = G
        bggr[1, 1, :] = r  # [1, 1] = R

        # 180도 회전 augmentation (구조 flip + B/R 교환)
        if self.augment and np.random.rand() < 0.5:
            struct = np.flip(struct, axis=(1, 2)).copy()  # H, W 모두 flip
            # B/R 교환: BGGR에서 [0,0]과 [1,1] swap
            bggr_aug = bggr.copy()
            bggr_aug[0, 0, :] = bggr[1, 1, :]  # B ← R
            bggr_aug[1, 1, :] = bggr[0, 0, :]  # R ← B
            bggr = bggr_aug

        return torch.from_numpy(struct), torch.from_numpy(bggr)


def create_dataloaders(cfg):
    """
    config dict로부터 train/val DataLoader 쌍 생성

    Returns: (train_loader, val_loader)
    """
    # seed 고정 (재현성)
    seed = cfg.get("seed", 42)
    g = torch.Generator()
    g.manual_seed(seed)

    # 데이터셋 생성 (augment=False, train split할 때 따로 제어)
    full_dataset = CRDataset(cfg["data"], augment=False)

    # train/val split
    train_ratio = cfg["data"].get("train_ratio", 0.95)
    train_size = int(len(full_dataset) * train_ratio)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=g
    )

    # augmentation은 train set에만 적용
    # (원본 dataset에 augment flag가 있으므로 별도로 처리)
    train_dataset_aug = CRDataset(cfg["data"], augment=True)
    train_indices = train_dataset.indices
    class AugmentedSubset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

    train_dataset_with_aug = AugmentedSubset(train_dataset_aug, train_indices)

    # DataLoader 생성
    batch_size = cfg["data"].get("batch_size", 64)
    num_workers = cfg["data"].get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset_with_aug,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
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
