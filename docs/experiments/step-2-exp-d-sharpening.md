# Step 2 - Experiment D: DINO Sharpening

## Purpose
Test if DINO-style sharpening (centering + temperature) helps teacherless training.

## Configuration
- Model: Teacherless (stop-grad) + Rate + Sharpening
- Sharpening: temp=0.04, center_momentum=0.9
- λ_rate: 0.03
- LR schedule: Cosine with 2-epoch warmup
- Dataset: CIFAR-10 rotation (20k subset)
- Epochs: 10

## Results

| Metric | Value | Gate |
|--------|-------|------|
| linacc_test | 27.35% | ❌ (<30%) |
| r2_test | 0.21 | ✅ (>0.1) |
| leakage_linacc | 33.7% | - |
| gate_pass | False | ❌ |

## Observations
1. **Sharpening degraded performance** compared to pure stop-grad (30.0% → 27.35%)
2. Loss went very negative (-10.2) - sharpening + rate may conflict
3. Performance unstable in mid-training (epoch 4: 20.4%)

## Analysis
The DINO sharpening applies softmax which normalizes outputs to sum to 1. This conflicts with:
1. Cosine similarity loss (which expects normalized vectors, not probability distributions)
2. Coding rate (which expects unconstrained representations)

## W&B
- Run: [yigdlml4](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/yigdlml4)

## Next Steps
- Try sharpening with different loss function (e.g., cross-entropy instead of cosine)
- Try lower sharpening temperature
- Try sharpening on predictor output only, not target

