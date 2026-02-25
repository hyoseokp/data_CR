import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

b0 = np.load('C:/Users/admin/phs/data_CR/code/CR_recon/dataset/bayer/bayer_0.npy', mmap_mode='r')
print('Shape:', b0.shape, 'Dtype:', b0.dtype)
print('Raw min:', b0.min(), 'max:', b0.max())

sample = -b0[:100].astype(np.float32)
print('GT range: min', sample.min(), 'max', sample.max())

for (r,c), cn in [((0,0),'R'), ((0,1),'G'), ((1,1),'B')]:
    ch = sample[:, r, c, :]
    print(f'{cn}: min={ch.min():.4f} max={ch.max():.4f} mean={ch.mean():.4f} std={ch.std():.4f}')

s0 = sample[0]
r_ch = s0[0, 0]
print(f'R: min={r_ch.min():.4f} max={r_ch.max():.4f} median={np.median(r_ch):.4f}')
peak_idx = int(np.argmax(r_ch))
print(f'Peak idx={peak_idx} val={r_ch[peak_idx]:.4f}')
print(f'Peak/median ratio: {r_ch[peak_idx]/max(np.median(r_ch), 1e-8):.1f}x')

dr = np.diff(r_ch)
print(f'Deriv: min={dr.min():.6f} max={dr.max():.6f} std={dr.std():.6f}')

threshold = np.median(r_ch) + 2 * np.std(r_ch)
n_peak = int(np.sum(r_ch > threshold))
print(f'Peak bins: {n_peak}/{len(r_ch)} = {100*n_peak/len(r_ch):.1f}%')

print(f'All in [0,1]? min={sample.min():.4f} max={sample.max():.4f}')
out_of_range = int(np.sum((sample < 0) | (sample > 1)))
print(f'Out of [0,1]: {out_of_range}/{sample.size} = {100*out_of_range/sample.size:.2f}%')
sys.stdout.flush()
