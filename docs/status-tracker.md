# Status Tracker

Last updated: 2026-01-05

## Current Focus

**Active Step**: Step 2 - Teacherless Rate (Exp A & B baseline-lite complete)  
**Next Step**: Run remote baselines, tune Exp B, run Exp C-E

## Milestone Status

| Milestone | Status | Progress | Notes |
|-----------|--------|----------|-------|
| M1 (Baseline) | 🟡 In Progress | 95% | Paper-aligned CIFAR-100 run in progress |
| M2 (Teacherless) | 🟡 In Progress | 60% | Exp A ✓, Exp B needs tuning |
| M3 (CRATE) | ⚪ Not Started | 0% | |
| M4 (ToST) | ⚪ Not Started | 0% | |
| M5 (Streaming) | ⚪ Not Started | 0% | |
| M6 (Local) | ⚪ Not Started | 0% | |

## Step Status

| Step | Status | Blocking Issues |
|------|--------|-----------------|
| Step 0 - Repo Setup | ✅ Complete | — |
| Step 1 - Baseline Repro | 🟡 In Progress | Paper-aligned run in progress; eval pending |
| Step 2 - Teacherless Rate | 🟡 In Progress | Exp A passing, Exp B needs tuning |
| Step 3 - CRATE Integration | ⚪ Pending | |
| Step 4 - ToST Streaming | ⚪ Pending | |
| Step 5 - Streaming Continual | ⚪ Pending | |
| Step 6 - Local Learning | ⚪ Pending | |

## Recent Work

### 2026-01-05
- **Step 2 experiments completed (local)**:
  - Fixed SSL/Zscaler with `truststore` package in train.py
  - **Exp A (EMA+Rate)**: Smoke ✓, Quick ✓, Baseline-lite ✓
    - linacc: 34.1%, r2: 0.302 (passes all gates)
    - W&B: [f1u4hzpj](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/f1u4hzpj)
  - **Exp B (StopGrad)**: Smoke ✓, Quick ✓, Baseline-lite ✓ (with issues)
    - linacc: 27.8%, r2: 0.144 (fails linacc gate)
    - Accuracy degraded in later epochs; needs tuning
    - W&B: [khgtp6bt](https://wandb.ai/kaikun213/seq-jepa-streaming/runs/khgtp6bt)
- **Key finding**: Coding rate prevents collapse, but stop-grad less stable than EMA
- **Created experiment logs**: `docs/experiments/step-2-exp-a-ema-rate.md`, `step-2-exp-b-stopgrad.md`
- **Previous session**: train.py integration, module creation
- **Remaining**: Run remote baselines, tune Exp B, run Exp C-E

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

- Baseline performance far below paper; need longer training and paper-aligned probe evaluation

## Notes for Agents

When updating this file:
1. Update the "Last updated" date
2. Adjust milestone/step status as work progresses
3. Add entries to "Recent Work" with date headers
4. Document any blockers in "Active Blockers"
