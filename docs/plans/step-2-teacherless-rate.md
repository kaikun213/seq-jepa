# Step 2 - Teacherless + coding-rate regularizer

**Status**: 🔄 Implementation in progress  
**Detailed plan**: See `.cursor/plans/teacherless_rate_reduction_*.plan.md`

**Key references**:
- Learning Diverse and Discriminative Representations via the Principle of Maximum Code Rate Reduction, 2020 ([arXiv:2006.08558](https://arxiv.org/abs/2006.08558)) - MCR² framework
- Simplifying DINO via Coding Rate Regularization, 2025 ([arXiv:2502.10385](https://arxiv.org/pdf/2502.10385)) - JEPA-like training without teacher using coding-rate term

## Goal
- Reduce dependence on EMA teacher without collapse.

## Experiments (progressive risk)
- **Experiment A**: Keep EMA, add coding-rate regularizer (lowest risk)
- **Experiment B**: Remove EMA, use stop-grad target (medium risk)
- **Experiment C**: Symmetric prediction (higher risk)
- **Experiment D**: DINO-style sharpening (subexperiment)
- **Experiment E**: CRATE components (subexperiment)

## Implementation Status

### ✅ Completed
- `experiments/models_teacherless.py` - Wrapper model with teacherless/rate-loss options
- `experiments/losses/coding_rate.py` - MCR²-based log-det rate loss
- `experiments/eval/group_probes.py` - Multi-factor action probes
- `experiments/eval/subspace_metrics.py` - PCA-based subspace analysis
- `experiments/eval/inv_eq_metrics.py` - Direct inv/eq error metrics
- `experiments/eval/viz.py` - W&B visualizations
- `experiments/datasets_mnist_affine.py` - MNIST controlled transforms
- `experiments/datasets_cifar_affine.py` - CIFAR controlled transforms
- `configs/step2/quick/exp_a_*.yaml` - Experiment A local configs (smoke/quick/baseline)
- `configs/step2/quick/exp_b_*.yaml` - Experiment B local configs (smoke/quick/baseline)
- `configs/step2/remote/exp_a_*.yaml` - Experiment A remote config
- `configs/step2/remote/exp_b_*.yaml` - Experiment B remote config

### ✅ Completed (continued)
- Integrate new losses and eval into `train.py` with config-driven flags
  - Uses `SeqJEPA_Teacherless` when `loss.rate_loss_enabled`, `model.teacherless`, `model.sharpening_enabled`, or `model.symmetric` are true
  - Falls back to upstream `SeqJEPA_Transforms` otherwise

### 🔄 In Progress
- Run Experiment A smoke → quick → baseline-lite → remote
- Run Experiment B smoke → quick → baseline-lite → remote
- Experiments C, D, E configs and runs

## Loss
```
L_pred = 1 - cos(z_hat_{t+1}, sg(z_{t+1}))
L_rate = -R(Z)
L_total = L_pred + lambda_rate * L_rate
```

R(Z) (expansion / anti-collapse) - based on MCR²:
```
C = (Z^T Z) / B
R(Z) = log det(I + alpha * C)
```

## Config Flags (new)
```yaml
model:
  teacherless: false          # Use stop-grad instead of EMA
loss:
  rate_loss_enabled: false    # Enable coding-rate loss
  lambda_rate: 0.01           # Weight
  rate_target: agg_out        # agg_out, z_t, or both
eval:
  subspace_enabled: false     # Run subspace metrics
```

## Hyperparameters
- lambda_rate: start 0.01, sweep 1e-3 to 1e-1
- alpha: 1.0 (tie to feature scale)

## Next Steps
1. ~~Modify `train.py` to use `SeqJEPA_Teacherless` when config flags present~~ ✅ Done
2. Run Experiment A smoke test to verify integration
3. Continue with quick → baseline-lite → remote progression
