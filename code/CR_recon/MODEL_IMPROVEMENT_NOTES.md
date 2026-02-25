# MetaSpec_DiffractionNet + AdaptiveSpectralLoss 개선 포인트

## 현재 상태 (Epoch 49/500, val_loss=1.285e-3)
- 매 epoch best 갱신 중, 학습 정상 진행
- loss weight: w_mse=2, w_rel=0.1, w_grad=100 (live) → 기여비 ~1:1:1 균형

---

## 잘 되어있는 것
- CNN backbone + FourierMixing + cross-attention decoder 구조 자체는 탄탄함
- Loss 3-term 구성 (MSE + RelMSE + GradMSE) + live weight 조절 가능
- Circular padding으로 주기적 경계조건 반영
- Sinusoidal wavelength embedding으로 스펙트럼 연속성 inductive bias

---

## 개선 가능한 포인트 (중요도순)

### 1. Cross-Attention에서 memory에 LayerNorm 없음 (중요도: 높음)
```python
# 현재: q만 norm하고 memory는 raw
x = self.norm_cross(q)
x2, _ = self.cross_attn(x, memory, memory, need_weights=False)
```
Q는 정규화하는데 K/V(memory)는 안 함. 학습 안정성에 영향.
일반적인 transformer decoder는 encoder output에도 LayerNorm을 적용함.

### 2. FiLM modulation이 additive만 있음 (중요도: 중간)
```python
# 현재: 단순 덧셈
q = q + film.unsqueeze(0)
```
FiLM 본래 설계: `gamma * x + beta` (scale + shift).
현재는 shift(beta)만 있고 scale(gamma)이 없음.
구조물에 따라 특정 파장 쿼리를 억제/증폭하는 능력이 부족.

### 3. Decoder output → Head가 단순함 (중요도: 중간)
```python
# 현재: 4개 채널(RGGB)을 하나의 Linear로 동시 예측
self.head = nn.Sequential(
    nn.Linear(d_model, d_model), nn.SiLU(),
    nn.Dropout(0.05),
    nn.Linear(d_model, 4),  # 4개 pixel을 공유 weight로
)
```
R, G1, G2, B 4개 디텍터가 서로 다른 위치에 있는데, 같은 head를 공유함.
각 채널별로 독립적인 head (또는 최소한 채널별 bias)가 있으면 더 좋음.

### 4. grad_clip=0.5가 보수적일 수 있음 (중요도: 중간)
현재 loss가 ~1.3e-3으로 작아서 gradient도 작음.
0.5 clip이 여전히 유효한지 모니터링 필요.
수렴이 느려지면 1.0으로 올려볼 가치 있음.

### 5. Loss에서 2차 미분 추가 가능 (중요도: 낮음~중간)
```python
# 현재: 1차 미분만
dp = pred[..., 1:] - pred[..., :-1]
```
2차 미분 `d2p = dp[..., 1:] - dp[..., :-1]`을 추가하면
peak의 곡률(뾰족함)을 직접 학습.
단, 지금은 1차로도 충분히 내려가고 있어서 수렴 후 시도.

### 6. Dropout 위치 (중요도: 낮음)
```python
# Decoder FFN에 dropout이 2번
self.ff = nn.Sequential(
    nn.Linear(d_model, hidden),
    nn.GELU(),
    nn.Dropout(dropout),        # 여기
    nn.Linear(hidden, d_model),
    nn.Dropout(dropout),        # 그리고 여기
)
```
training data가 30만개로 충분히 많아서 dec_dropout=0.03은 거의 의미 없는 수준.
제거해도 됨.

### 7. Spatial tokens 17개 한계 (중요도: 수렴 후 판단)
301채널 출력 대비 17개 KV 토큰이 빠듯할 수 있음.
수렴 후 plateau 치면 Stage3(8x8=65 tokens)로 확장 고려.

---

## 적용 우선순위

| 상태 | 설명 |
|------|------|
| 지금 당장 바꿔야 하는 것 | 없음. 학습이 잘 내려가고 있음 |
| 수렴 정체 시 1순위 | 1. memory LayerNorm 추가, 2. FiLM에 scale 추가, 3. grad_clip 1.0으로 |
| 다음 실험 후보 | 4. 채널별 head 분리, 5. spatial tokens 확장, 6. 2차 미분 loss |
