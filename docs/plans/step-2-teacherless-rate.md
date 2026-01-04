# Step 2 - Teacherless + coding-rate regularizer

**Key references**:
- Learning Diverse and Discriminative Representations via the Principle of Maximum Code Rate Reduction, 2020 ([arXiv:2006.08558](https://arxiv.org/abs/2006.08558)) - MCR² framework
- Simplifying DINO via Coding Rate Regularization, 2025 ([arXiv:2502.10385](https://arxiv.org/pdf/2502.10385)) - JEPA-like training without teacher using coding-rate term

## Goal
- Reduce dependence on EMA teacher without collapse.

## Options (least to most risky)
- Option A: keep EMA, add coding-rate regularizer.
- Option B: remove EMA, use stop-grad target from same encoder.
- Option C: symmetric prediction (forward and reverse).

## Loss
```
L_pred = 1 - cos(z_hat_{t+1}, sg(z_{t+1}))
L_rate = -R(Z)
L_total = L_pred + lambda_rate * L_rate
```

R(Z) example (expansion / anti-collapse) - based on MCR²:
```
C = (Z^T Z) / B
R(Z) = log det(I + alpha * C)
```

Apply first on z_AGG, optionally also on z_t with caution.

## Hyperparameters
- lambda_rate: start 1e-3, sweep to 1e-1
- alpha: tie to feature scale, tune as needed

## Diagnostics
- Feature std per dim and eigen spectrum.
- Mean cosine similarity across samples.
- Loss breakdowns.

## Fallbacks if collapse
- Keep EMA and slow its update.
- Add variance/cov penalties (VICReg-style) as stabilizer.
