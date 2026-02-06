"""
데이터 전처리: 원본 데이터를 Bayer 패턴으로 변환 및 저장
(3, 301) spectra → (2, 2, 30) Bayer 패턴

이 스크립트는:
1. 원본 spectra 파일 로드
2. 유효한 샘플 필터링 (zero 제거)
3. Bayer 패턴 변환 (3, 301) → (2, 2, 30)
4. 180도 회전 버전 생성 (R과 B 교환)
5. 새로운 .npy 파일로 저장
"""
import numpy as np
from pathlib import Path
import time


def convert_to_bayer(spectra, out_len=30):
    """
    (M, 3, 301) → (M, 2, 2, out_len) Bayer 패턴 변환

    입력: spectra (M, 3, 301) = [R, G, B] × 301 파장
    출력: bayer (M, 2, 2, out_len) = [[R, G], [G, B]] × out_len 파장
    """
    M = len(spectra)
    bayer = np.zeros((M, 2, 2, out_len), dtype=np.float32)

    # 다운샘플 indices
    indices = np.linspace(0, 300, out_len, dtype=int)

    print(f"  Bayer 변환 중 ({M} 샘플)...")
    for i in range(M):
        if (i + 1) % max(1, M // 10) == 0:
            print(f"    {i + 1}/{M} 완료")

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


def create_rotated_bayer(bayer):
    """
    Bayer 패턴에서 180도 회전 = R과 B 위치 교환

    원본: [0,0]=R, [0,1]=G, [1,0]=G, [1,1]=B
    회전: [0,0]=B, [0,1]=G, [1,0]=G, [1,1]=R
    """
    M = len(bayer)
    bayer_rotated = np.zeros_like(bayer)

    print(f"  180도 회전 생성 중 ({M} 샘플)...")
    bayer_rotated[:, 0, 0, :] = bayer[:, 1, 1, :]  # R@(0,0) ← B@(1,1)
    bayer_rotated[:, 0, 1, :] = bayer[:, 0, 1, :]  # G@(0,1) = G@(0,1)
    bayer_rotated[:, 1, 0, :] = bayer[:, 1, 0, :]  # G@(1,0) = G@(1,0)
    bayer_rotated[:, 1, 1, :] = bayer[:, 0, 0, :]  # B@(1,1) ← R@(0,0)

    return bayer_rotated


def main():
    print("\n" + "="*80)
    print("DATA PREPROCESSING: Spectra → Bayer Pattern + Struct Filtering")
    print("="*80)

    # 경로 계산
    # preprocess_data.py: plan-dl-cr-dashboard/code/CR_recon/
    cr_recon_dir = Path(__file__).parent  # CR_recon/
    plan_dir = cr_recon_dir.parent.parent  # plan-dl-cr-dashboard/

    data_dir = plan_dir / "data_CR-main"
    output_dir = cr_recon_dir / "dataset" / "bayer"

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"데이터 경로: {data_dir}")
    print(f"출력 디렉토리: {output_dir}\n")

    # 처리할 파일들
    struct_files = [
        data_dir / "binary_dataset_128_0.npy",
        data_dir / "binary_dataset_128_1.npy"
    ]
    spectra_files = [
        data_dir / "spectra_latest_0.npy",
        data_dir / "spectra_latest_1.npy"
    ]

    for file_idx, (struct_file, spec_file) in enumerate(zip(struct_files, spectra_files)):
        if not spec_file.exists() or not struct_file.exists():
            print(f"\n✗ 파일 없음:")
            if not spec_file.exists():
                print(f"  {spec_file}")
            if not struct_file.exists():
                print(f"  {struct_file}")
            continue

        print(f"\n[파일 {file_idx}] {spec_file.name} + {struct_file.name}")
        print("-" * 80)

        # 1. 원본 데이터 로드
        print("  원본 데이터 로드 중...")
        start_time = time.time()
        struct = np.load(str(struct_file))
        spectra = np.load(str(spec_file))
        print(f"    Struct Shape: {struct.shape}")
        print(f"    Spectra Shape: {spectra.shape}")
        print(f"    로드 시간: {time.time() - start_time:.2f}초")

        # 2. 유효한 샘플 필터링 (spectra 기준)
        print("  유효 샘플 필터링 중...")
        valid_mask = np.any(spectra != 0, axis=(1, 2))
        struct_valid = struct[valid_mask]
        spectra_valid = spectra[valid_mask]
        n_removed = len(spectra) - len(spectra_valid)
        print(f"    원본: {len(spectra)} → 유효: {len(spectra_valid)} (제거: {n_removed})")
        print(f"    Struct도 함께 필터링됨: {struct.shape} → {struct_valid.shape}")

        # 3. Bayer 패턴 변환
        print("  Bayer 패턴 변환 중...")
        start_time = time.time()
        bayer = convert_to_bayer(spectra_valid, out_len=30)
        print(f"    출력 Shape: {bayer.shape}")
        print(f"    변환 시간: {time.time() - start_time:.2f}초")

        # 4. 180도 회전 버전 생성
        print("  180도 회전 버전 생성 중...")
        start_time = time.time()
        bayer_rotated = create_rotated_bayer(bayer)
        print(f"    회전 Shape: {bayer_rotated.shape}")
        print(f"    생성 시간: {time.time() - start_time:.2f}초")

        # 5. 저장
        print("  파일 저장 중...")
        start_time = time.time()

        # 필터링된 Struct
        output_file_struct = output_dir / f"struct_{file_idx}.npy"
        np.save(str(output_file_struct), struct_valid)
        print(f"    [OK] {output_file_struct.name} ({struct_valid.nbytes / (1024**3):.2f} GB)")

        # 원본 Bayer
        output_file_bayer = output_dir / f"bayer_{file_idx}.npy"
        np.save(str(output_file_bayer), bayer)
        print(f"    [OK] {output_file_bayer.name} ({bayer.nbytes / (1024**3):.2f} GB)")

        # 회전 Bayer
        output_file_rotated = output_dir / f"bayer_rotated_{file_idx}.npy"
        np.save(str(output_file_rotated), bayer_rotated)
        print(f"    [OK] {output_file_rotated.name} ({bayer_rotated.nbytes / (1024**3):.2f} GB)")

        print(f"    저장 시간: {time.time() - start_time:.2f}초")

        # 검증
        print("  검증:")
        spec_sample = bayer[0]
        is_g_same = np.allclose(spec_sample[0, 1], spec_sample[1, 0])
        is_rb_diff = not np.allclose(spec_sample[0, 0], spec_sample[1, 1])
        print(f"    G[0,1] == G[1,0]? {is_g_same} {'[OK]' if is_g_same else '[FAIL]'}")
        print(f"    R != B? {is_rb_diff} {'[OK]' if is_rb_diff else '[FAIL]'}")

    print("\n" + "="*80)
    print("[완료] 데이터 전처리 완료!")
    print(f"출력 디렉토리: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
