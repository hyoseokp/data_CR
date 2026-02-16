"""
최적 batch_size 탐색: 1 epoch 소요 시간이 최소가 되는 batch_size를 찾는다.
GPU 메모리 한계 내에서 throughput(samples/sec)이 최대인 지점을 탐색.
"""
import gc
import time
import torch
import numpy as np
from models import get_model
from losses import get_loss
from utils import load_config

def benchmark_batch_size(bs, model, loss_fn, device, in_shape=(1, 128, 128), out_shape=(2, 2, 301), n_iters=20, warmup=5):
    """주어진 batch_size로 forward+backward 시간 측정, samples/sec 반환"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")

    x = torch.randn(bs, *in_shape, device=device)
    y = torch.randn(bs, *out_shape, device=device)

    # warmup
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pred = model(x)
            loss = loss_fn(pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pred = model(x)
            loss = loss_fn(pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    samples_per_sec = (bs * n_iters) / elapsed
    sec_per_iter = elapsed / n_iters
    return samples_per_sec, sec_per_iter

def main():
    cfg = load_config("configs/default.yaml")
    device = torch.device("cuda")

    model_params = cfg["model"]["params"]
    model = get_model(cfg["model"]["name"], **model_params).to(device)
    loss_fn = get_loss(cfg["loss"]["name"], **cfg["loss"].get("params", {})).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']['name']} ({param_count:,} params)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # 총 train 샘플 수 (augment_180 적용 시)
    total_samples = 287390  # from previous log
    print(f"Total train samples per epoch: {total_samples:,}")
    print()

    # batch sizes to test (power of 2)
    candidates = [64, 128, 256, 512, 768, 1024]

    results = []
    print(f"{'BS':>6} | {'samples/s':>10} | {'s/iter':>8} | {'iters/epoch':>11} | {'epoch_time':>10} | {'VRAM_used':>10}")
    print("-" * 75)

    for bs in candidates:
        # 메모리 정리
        gc.collect()
        torch.cuda.empty_cache()

        try:
            sps, spi = benchmark_batch_size(bs, model, loss_fn, device)
            iters_per_epoch = int(np.ceil(total_samples / bs))
            epoch_time = iters_per_epoch * spi
            vram = torch.cuda.max_memory_allocated() / 1024**3

            results.append((bs, sps, spi, iters_per_epoch, epoch_time, vram))
            print(f"{bs:>6} | {sps:>10.1f} | {spi:>8.4f} | {iters_per_epoch:>11} | {epoch_time:>8.1f}s | {vram:>8.1f} GB")

            # 메모리 카운터 리셋
            torch.cuda.reset_peak_memory_stats()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{bs:>6} | {'OOM':>10} | {'--':>8} | {'--':>11} | {'--':>10} | {'--':>10}")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            else:
                raise

    print()
    if results:
        best = min(results, key=lambda r: r[4])  # epoch_time 최소
        print(f"==> Best batch_size = {best[0]}  (epoch ~{best[4]:.0f}s, {best[1]:.0f} samples/s, VRAM {best[5]:.1f} GB)")

if __name__ == "__main__":
    main()
