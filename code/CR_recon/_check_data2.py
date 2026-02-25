import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

b0 = np.load('C:/Users/admin/phs/data_CR/code/CR_recon/dataset/bayer/bayer_0.npy', mmap_mode='r')
b1 = np.load('C:/Users/admin/phs/data_CR/code/CR_recon/dataset/bayer/bayer_1.npy', mmap_mode='r')
all_data = -np.concatenate([b0, b1], axis=0).astype(np.float32)
print(f'Total samples: {len(all_data)}, shape: {all_data.shape}')
print(f'Global range: [{all_data.min():.4f}, {all_data.max():.4f}]')

# Overall statistics
print(f'\n=== Global channel stats ===')
for (r,c), cn in [((0,0),'R'), ((0,1),'G1'), ((1,0),'G2'), ((1,1),'B')]:
    ch = all_data[:, r, c, :]
    print(f'{cn}: range=[{ch.min():.4f}, {ch.max():.4f}] mean={ch.mean():.4f} std={ch.std():.4f}')

# Sigmoid output range: [0, 1], but GT range is [0.07, 0.49]
# This means sigmoid needs to output ~0.07-0.49, i.e., pre-sigmoid logits ~[-2.6, -0.04]
# The dynamic range model needs to resolve: 0.49 - 0.07 = 0.42
# MSE error 0.014 (sqrt(2e-4)) is ~3.3% of this dynamic range

print(f'\n=== Dynamic range analysis ===')
dynamic_range = all_data.max() - all_data.min()
print(f'Dynamic range: {dynamic_range:.4f}')
mse_err = np.sqrt(2.05e-4)
print(f'Current MSE RMSE: {mse_err:.4f} = {100*mse_err/dynamic_range:.1f}% of dynamic range')

# Peak analysis across many samples
print(f'\n=== Peak structure analysis (100 samples) ===')
for (r,c), cn in [((0,0),'R'), ((1,1),'B')]:
    peak_heights = []
    peak_widths = []
    peak_contrasts = []
    for i in range(min(1000, len(all_data))):
        ch = all_data[i, r, c, :]
        baseline = np.median(ch)
        peak_idx = np.argmax(ch)
        peak_val = ch[peak_idx]
        contrast = peak_val - baseline
        peak_heights.append(peak_val)
        peak_contrasts.append(contrast)
        # Peak width (FWHM approximation)
        half_max = baseline + contrast / 2
        above = ch > half_max
        if above.any():
            peak_widths.append(np.sum(above))

    peak_heights = np.array(peak_heights)
    peak_contrasts = np.array(peak_contrasts)
    peak_widths = np.array(peak_widths)
    print(f'{cn}: peak_contrast mean={peak_contrasts.mean():.4f} std={peak_contrasts.std():.4f}')
    print(f'{cn}: peak_width(FWHM) mean={peak_widths.mean():.1f} std={peak_widths.std():.1f} bins')
    print(f'{cn}: contrast/MSE_err = {peak_contrasts.mean()/mse_err:.1f}x')

# High frequency content: what fraction of total energy is in derivatives
print(f'\n=== Spectral smoothness ===')
for (r,c), cn in [((0,0),'R'), ((1,1),'B')]:
    ch = all_data[:1000, r, c, :]
    deriv = np.diff(ch, axis=-1)
    deriv2 = np.diff(deriv, axis=-1)
    print(f'{cn}: |deriv| mean={np.abs(deriv).mean():.6f} max={np.abs(deriv).max():.6f}')
    print(f'{cn}: |deriv2| mean={np.abs(deriv2).mean():.6f} max={np.abs(deriv2).max():.6f}')
    print(f'{cn}: value_std={ch.std():.4f} deriv_std={deriv.std():.6f} ratio={ch.std()/deriv.std():.1f}x')

sys.stdout.flush()
