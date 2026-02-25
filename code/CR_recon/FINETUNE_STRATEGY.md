# Fine-Tuning 전략: 소규모 데이터 (100개) 적용 가이드

## 배경
- Pretrain: 30만개 시뮬레이션 데이터로 500 epoch 학습 완료
- Fine-tune: 100개 소규모 데이터로 특정 조건에 맞게 보정
- 모델 파라미터: 6.45M → 100개로 전체 학습 시 overfitting 확실

---

## 전략 1: Output Head만 학습 (추천 1순위)

```
[CNN Encoder]        → freeze
[FourierMixing]      → freeze
[SpatialPool]        → freeze
[WavelengthEmbed]    → freeze
[FiLM]               → freeze
[Decoder Block x4]   → freeze
[Output Head]        → 학습  (Linear 256->256->4 + spec_bias)
```

- 학습 파라미터: ~66K (전체의 1%)
- 100개로도 충분
- CNN이 배운 구조 인식 능력 보존
- spec_bias (1,4,301) = 1,204 파라미터가 스펙트럼 오프셋 보정에 핵심

## 전략 2: Decoder + Head 학습 (추천 2순위)

```
[CNN Encoder]        → freeze
[FourierMixing]      → freeze
[SpatialPool]        → freeze
[WavelengthEmbed]    → freeze 또는 학습
[FiLM]               → 학습
[Decoder Block x4]   → 학습
[Output Head]        → 학습
```

- 학습 파라미터: ~3.5M
- 100개로는 빠듯하지만 strong augmentation으로 보완 가능
- CNN의 공간 특징 추출은 보존하면서 스펙트럼 매핑만 재학습

## 전략 3: 전체 학습 + 극소 lr (비추)

- 전체 모델을 매우 낮은 lr로 몇 epoch만
- catastrophic forgetting 위험 높음

---

## 하이퍼파라미터 권장

| 항목 | 전략 1 (Head만) | 전략 2 (Decoder+Head) |
|------|----------------|---------------------|
| lr | 1e-4 ~ 5e-4 | 1e-5 ~ 5e-5 |
| epochs | 50~100 | 20~50 |
| batch_size | 8~16 | 8~16 |
| weight_decay | 0.01 | 0.01 |
| warmup | 불필요 | 5 epoch |
| augmentation | 180 회전 | 180 회전 + noise |
| scheduler | cosine | cosine |
| early stopping | val_loss 10 epoch 미개선 시 | val_loss 5 epoch 미개선 시 |

---

## Overfitting 방지 핵심

1. **Train/Val split**: 100개 중 80/20 또는 90/10. 반드시 val set 유지
2. **Data augmentation**: 180 회전 (현재 augment_180), 좌우/상하 flip 추가 가능 → 100개 → 400~800개
3. **Early stopping**: val_loss 모니터링 필수
4. **Dropout 유지 또는 증가**: head_dropout=0.1~0.2로 올리기
5. **Weight decay 증가**: 0.01~0.05

---

## 구현 방법

### Config 예시

```yaml
# configs/finetune_diffraction.yaml
training:
  epochs: 100
  lr: 0.0003
  weight_decay: 0.01
  grad_clip: 1.0
  warmup_ratio: 0.0

  # fine-tune 설정
  resume: "outputs/cnn_diffraction_best.pt"
  freeze_encoder: true        # CNN + FourierMixing freeze
  freeze_decoder: false       # 전략에 따라 true/false
```

### trainer.py freeze 로직

```python
# freeze 예시
if cfg["training"].get("freeze_encoder", False):
    for name, param in self.model.named_parameters():
        if any(k in name for k in ['stem', 'stage1', 'stage2', 'stage3',
                                      'stage4', 'fourier_mix', 'spatial_pool']):
            param.requires_grad = False
```

---

## 데이터 성격별 최적 전략

| 데이터 성격 | 최적 전략 |
|------------|----------|
| 실측 데이터 (시뮬레이션->실제 보정) | 전략 1 (Head만) — sim-to-real gap은 주로 스펙트럼 오프셋/스케일 |
| 특수 구조 (학습 분포 밖의 새로운 패턴) | 전략 2 (Decoder+Head) — 새로운 구조-스펙트럼 매핑 필요 |
| 고정밀 레이블 (기존과 같은 분포, 더 정확한 GT) | 전략 1 (Head만) + spec_bias 위주 |
