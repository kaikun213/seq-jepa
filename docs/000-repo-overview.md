# Repository Overview: seq-JEPA Implementation

## High-Level Structure

This repository implements the **seq-JEPA** (Sequential Joint-Embedding Predictive Architecture) paper, which learns invariant and equivariant representations from sequential visual observations with action conditioning.

### Directory Structure

```
seq-jepa/
├── seq-jepa/              # Upstream code snapshot (DO NOT EDIT)
│   ├── src/               # Core models, training, utilities
│   ├── scripts/           # Original entrypoints
│   └── data/              # Local datasets
├── experiments/          # Local modifications/extensions
├── configs/              # YAML configuration files
├── train.py              # Wrapper entrypoint (use this)
├── docs/                 # Planning and documentation
└── runs/                 # Experiment outputs
```

## Core Architecture

### Model: `SeqJEPA_Transforms`

**Location**: `seq-jepa/src/models.py` (lines 193-388) and `experiments/models_seqjepa.py`

**Key Components**:

1. **Encoder** (`self.encoder`): ResNet18 backbone
   - Encodes individual frames: `z_t = encoder(x_t)`
   - Output dimension: `res_out_dim = 512` (ResNet18 features)

2. **Action Conditioning** (`self.action_proj`): Optional learned action embeddings
   - Projects action latents to `action_projdim` (default: 128)
   - Creates relative action encodings between frames

3. **Transformer Aggregator** (`self.transformer_encoder`):
   - Takes sequence of tokens: `[AGG_token, z_1+action_1, z_2+action_2, ...]`
   - Outputs aggregated representation from `[AGG]` token: `z_AGG`

4. **Predictor** (`self.predictor`):
   - Takes `(z_AGG, action_M)` and predicts `z_hat_{M+1}`
   - MLP with hidden dimension `pred_hidden` (default: 1024)

5. **EMA Target Encoder** (`self.target_encoder`): Optional
   - Exponential moving average of encoder weights
   - Used to generate stop-gradient targets: `z_{M+1} = target_encoder(x_{M+1}).detach()`

### Forward Pass Flow

```python
# Input: x_obs (B, seq_len, C, H, W), x_pred (B, C, H, W)
#        act_lat_obs (B, seq_len, act_dim), act_lat_pred (B, act_dim)

1. Encode observations: z_obs = encoder(x_obs)  # (B, seq_len, 512)
2. Encode prediction target: z_pred = target_encoder(x_pred).detach()  # (B, 512)

3. Action conditioning:
   - Compute relative actions: rel_actions = actions[1:] - actions[:-1]
   - Project: act_emb = action_proj(rel_actions)  # (B, seq_len, 128)
   - Concatenate: tokens = concat([z_obs, act_emb], dim=-1)  # (B, seq_len, 640)

4. Transformer aggregation:
   - Add [AGG] token: x = concat([AGG_token, tokens], dim=1)
   - Transform: x = transformer_encoder(x)
   - Extract: z_AGG = x[:, 0]  # (B, 640)

5. Prediction:
   - Condition on final action: z_AGG_cond = concat([z_AGG, action_proj(act_lat_pred)])
   - Predict: z_hat = predictor(z_AGG_cond)  # (B, 512)

6. Loss: L = 1 - cosine_similarity(z_hat, z_pred).mean()
```

### Training Loop

**Location**: `train.py` (lines 202-284 for training, 287-347 for evaluation)

**Key Steps**:
1. Forward pass through model → get `loss, agg_out, z1, z2`
2. Backward pass and optimizer step
3. Update EMA if enabled: `model.update_moving_average()`
4. Online probe training (linear classifier on `agg_out`, equivariance probe on `z1, z2`)

## Key Files

### Entry Points
- **`train.py`**: Main wrapper script (use this, not `seq-jepa/src/main_*.py`)
  - Loads config from YAML
  - Sets up model, data, optimizers
  - Runs training loop with online probes
  - Saves metrics to `runs/{name}/`

### Models
- **`seq-jepa/src/models.py`**: Original upstream models (preserve)
- **`experiments/models_seqjepa.py`**: Local copy of `SeqJEPA_Transforms` (can modify)

### Data
- **`experiments/datasets_rot.py`**: `CIFAR10RotationSequence` dataset
  - Creates sequences of rotated CIFAR-10 images
  - Actions are one-hot rotation angles: `[0, 90, 180, 270]`

### Configuration
- **`configs/quick/*.yaml`**: Experiment configs
  - `cifar10_rot_baseline.yaml`: Full baseline with EMA
  - `cifar10_rot_quick.yaml`: Quick test
  - `cifar10_rot_smoke.yaml`: Smoke test

## Where to Make Changes for Step 2 (Teacherless + Coding Rate)

### Option A: Keep EMA, Add Coding Rate Regularizer

**Files to modify**:
1. **`experiments/models_seqjepa.py`**:
   - Add `coding_rate_loss()` method to compute `R(Z) = log det(I + alpha * C)`
   - Modify `forward()` to return additional loss term
   - Add hyperparameters: `lambda_rate`, `alpha`

2. **`train.py`**:
   - Modify `train_one_epoch_local()` (line 202) to:
     - Extract `agg_out` from model forward
     - Compute coding rate loss: `L_rate = -R(agg_out)`
     - Combine: `L_total = L_pred + lambda_rate * L_rate`
   - Add config parameters: `model.coding_rate.lambda_rate`, `model.coding_rate.alpha`

**Key locations**:
```python
# In experiments/models_seqjepa.py, add method:
def coding_rate_loss(self, Z, alpha=1.0):
    """
    Compute coding rate: R(Z) = log det(I + alpha * C)
    where C = (Z^T Z) / B
    """
    B = Z.shape[0]
    C = (Z.T @ Z) / B
    R = torch.logdet(torch.eye(C.shape[0], device=Z.device) + alpha * C)
    return R

# In forward(), modify return:
loss_pred = 1 - self.criterion(pred_out, enc_pred_detached).mean()
loss_rate = -self.coding_rate_loss(agg_out, alpha=self.alpha)
loss = loss_pred + self.lambda_rate * loss_rate
return loss, agg_out, z1, z2
```

### Option B: Remove EMA, Use Stop-Grad from Same Encoder

**Files to modify**:
1. **`experiments/models_seqjepa.py`**:
   - Set `ema=False` in config or constructor
   - Modify `forward()` to use `self.encoder` instead of `self.target_encoder`
   - Still detach: `enc_pred_detached = self.encoder(x_pred).detach()`

2. **`train.py`**:
   - Remove EMA update call: `model.update_moving_average()` (line 248)
   - Ensure `model_cfg.get("ema", False)` is False

### Option C: Symmetric Prediction

**Files to modify**:
1. **`experiments/models_seqjepa.py`**:
   - Modify `forward()` to predict both forward and reverse
   - Compute bidirectional loss: `L_forward + L_reverse`

## Implementation Strategy

### Recommended Approach

1. **Create new model variant**: `SeqJEPA_Transforms_Teacherless` in `experiments/models_seqjepa.py`
   - Copy `SeqJEPA_Transforms` class
   - Add coding rate computation
   - Make EMA optional (already supported via `ema=False`)

2. **Extend training loop**: Modify `train_one_epoch_local()` in `train.py`
   - Add coding rate loss computation
   - Combine losses with configurable `lambda_rate`

3. **Add config parameters**: Update YAML configs
   ```yaml
   model:
     ema: false  # Option B: remove EMA
     coding_rate:
       enabled: true
       lambda_rate: 0.001  # Start small, sweep to 0.1
       alpha: 1.0  # Tune based on feature scale
   ```

4. **Add diagnostics**: Track feature statistics
   - Feature std per dimension
   - Eigen spectrum of covariance
   - Mean cosine similarity across samples

## Important Details

### Representation Expectations
- **`z_t`** (encoder outputs): Should be **equivariant** (changes with actions)
- **`z_AGG`** (aggregator output): Should be **invariant** (stable across sequences)
- Online probes verify this:
  - Linear probe on `agg_out` → classification accuracy (invariance)
  - Equivariance probe on `[z1, z2]` → action regression R² (equivariance)

### Action Conditioning
- Actions are **relative** between consecutive frames
- Last action in sequence is used to condition prediction
- If `learn_act_emb=True`, actions are projected through MLP

### EMA Mechanism
- Default decay: `0.996`
- Updated every step: `target_params = decay * target_params + (1-decay) * online_params`
- Prevents collapse by providing stable targets

### Loss Function
- **Prediction loss**: `L_pred = 1 - cosine_similarity(z_hat, z_target)`
- **Coding rate** (to add): `L_rate = -R(Z)` where `R(Z) = log det(I + alpha * C)`
- **Total**: `L_total = L_pred + lambda_rate * L_rate`

## Next Steps for Step 2

1. **Implement coding rate computation** in model
2. **Modify training loop** to include rate loss
3. **Create config** for teacherless variant
4. **Add diagnostics** for collapse detection
5. **Test with small `lambda_rate`** (1e-3), then sweep to 1e-1

## References

- Step 2 plan: `docs/012-step-2-teacherless-rate.md`
- Baseline architecture: `docs/003-architecture-baseline.md`
- MCR² paper: Learning Diverse and Discriminative Representations via the Principle of Maximum Code Rate Reduction (2020)
- DINO simplification: Simplifying DINO via Coding Rate Regularization (2025)

