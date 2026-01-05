# Status Tracker

Last updated: 2026-01-05

## Current Focus

**Active Step**: Step 2 - Remote experiments running  
**Next Step**: Compare EMA vs Teacherless results, then step-3 CRATE integration

## Active Remote Jobs

| Instance ID | Label | GPU | Status | W&B Run | Notes |
|-------------|-------|-----|--------|---------|-------|
| 29522248 | seqjepa-step2 | RTX 3060 | 🟡 Running (v1) | TBD | Old config, monitoring for recovery |
| 29527387 | seqjepa-step2-exp-a | RTX 4090/3090 | 🟢 Running (v2) | TBD | EMA + Rate, paper-aligned |
| 29527388 | seqjepa-step2-exp-b | RTX 4090/3090 | 🟢 Running (v2) | TBD | Teacherless + Rate, paper-aligned |

## Milestone Status

| Milestone | Status | Progress | Notes |
|-----------|--------|----------|-------|
| M1 (Baseline) | ✅ Complete | 100% | Paper run: 57.84% linacc, 0.739 R² |
| M2 (Teacherless) | 🟡 In Progress | 85% | Exp A-D ✓, subspace integrated, remote running |
| M3 (CRATE) | ⚪ Not Started | 0% | |
| M4 (ToST) | ⚪ Not Started | 0% | |
| M5 (Streaming) | ⚪ Not Started | 0% | |
| M6 (Local) | ⚪ Not Started | 0% | |

## Step Status

| Step | Status | Blocking Issues |
|------|--------|-----------------|
| Step 0 - Repo Setup | ✅ Complete | — |
| Step 1 - Baseline Repro | ✅ Complete | Paper run finished: [eg1gydas](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/eg1gydas) |
| Step 2 - Teacherless Rate | 🟡 In Progress | v2 running in parallel (29527387, 29527388) |
| Step 3 - CRATE Integration | ⚪ Pending | |
| Step 4 - ToST Streaming | ⚪ Pending | |
| Step 5 - Streaming Continual | ⚪ Pending | |
| Step 6 - Local Learning | ⚪ Pending | |

## Recent Work

### 2026-01-05
- **Step 1 paper baseline COMPLETE** ✅:
  - W&B: [eg1gydas](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/eg1gydas)
  - **linacc: 57.84%**, **R²: 0.739** (2000 epochs, 23.4 hours)
  - Leakage gaps positive (inv/eq separation confirmed)
  - Destroyed idle instance 29493608 (was still running after completion)
- **Step 2 experiments completed (local)**:
  - Fixed SSL/Zscaler with `truststore` package in train.py
  - **Exp A (EMA+Rate)**: Smoke ✓, Quick ✓, Baseline-lite ✓
    - linacc: 34.1%, r2: 0.302 (passes all gates)
    - W&B: [f1u4hzpj](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/f1u4hzpj)
  - **Exp B (StopGrad)**: Multiple tuning iterations
    - Original: linacc 27.8%, r2 0.144 (failed - degraded in later epochs)
    - High λ_rate (0.1): linacc 21.4% (worse - rate dominated)
    - **Tuned (warmup + cosine + λ=0.03)**: linacc 30.0%, r2 0.423 ✓
    - W&B: [sjd2550e](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/sjd2550e)
  - **MNIST Subspace Diagnostics**: Initial run with EMA+rate
    - 20 epochs, r2: 0.256-0.370 (equivariance learning ✓)
    - W&B: [7z37n2lu](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/7z37n2lu)
- **Key findings**:
  1. Coding rate prevents collapse in both EMA and stop-grad modes
  2. Stop-grad needs cosine LR + warmup for stability
  3. λ_rate=0.03 is optimal (0.01 too weak, 0.1 too strong)
- **Created experiment logs**: `step-2-exp-a-ema-rate.md`, `step-2-exp-b-stopgrad.md`, `step-2-mnist-subspace.md`
- **Exp C (Symmetric)**: linacc 33.3%, r2 0.42 ✅ (no clear benefit vs baseline)
  - W&B: [hj0seblj](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/hj0seblj)
- **Exp D (Sharpening)**: linacc 27.4%, r2 0.21 ❌ (sharpening hurts with cosine loss)
  - W&B: [yigdlml4](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/yigdlml4)
- **Subspace metrics integrated**: rank_eq_eff, rank_inv_eff, subspace_ev, etc. logged to W&B
- **MNIST Subspace (v2)**: r2 0.47 (strong equivariance learning)
  - W&B: [qganshbm](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/qganshbm)
- **Remote launch prepared**: 
  - `configs/remote/step2_exp_a_ema_rate.yaml` (30 epochs, EMA+Rate)
  - `configs/remote/step2_exp_b_teacherless.yaml` (30 epochs, Teacherless)
  - Launch: `scripts/vast/launch_step2.sh`
- **Decision**: Skip Exp C-E (sharpening, symmetric, CRATE) - teacherless works, keep simple
- **Config fix**: v1 remote run had divergent config (batch_size=128, lr=0.001, no weight_decay)
  - Fixed to paper defaults: batch_size=512, lr=0.0004, weight_decay=0.001, AdamW
- **v2 parallel launch**: Instances 29527387 (EMA), 29527388 (Teacherless) on RTX 4090/3090
- **GPU preference updated**: RTX 4090 primary, RTX 3090 backup
- **Remaining**: Monitor v2 runs, compare EMA vs Teacherless, then step-3 CRATE

### 2026-01-04
- Reorganized documentation structure into subdirectories
- Set up local git config for private email
- Ran clean Vast CIFAR-100 baseline; results far below paper targets
- Started paper-aligned CIFAR-100 run on Vast (long, eval pending)
- Updated CIFAR-100 gate config and metrics/gates guidance for remote comparisons
- Archived deprecated 10-epoch CIFAR-100 baseline artifacts and added local gate script/config
- Started local 200-epoch CIFAR-100 gate run (MPS)
- Interrupted local gate at epoch 110; updated gate to 30 epochs with new thresholds

### 2026-01-03
- Smoke and quick runs passing on CIFAR-10 rotations
- W&B integration working
- Created metrics-and-gates.md with threshold definitions

### 2026-01-02
- Initial repo setup complete
- Created experiments/ directory structure
- Added CIFAR-100 augmentation support

## Active Blockers

- None currently (Step 1 complete, Step 2 remote running)

## Notes for Agents

When updating this file:
1. Update the "Last updated" date
2. Adjust milestone/step status as work progresses
3. Add entries to "Recent Work" with date headers
4. Document any blockers in "Active Blockers"
