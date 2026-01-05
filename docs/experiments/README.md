# Experiment Logs

This directory tracks all experiments: what was tried, results, and decisions.
Agents must log every experiment run here and add it to the index below.

## Creating New Experiments

1. Copy the template: `docs/templates/experiment-template.md`
2. Rename to `step-N-experiment-name.md`
3. Fill in all sections, especially **Preliminary Analysis** (required)
4. Add to the index below

## Structure

- `step-1-baseline-*.md` - Baseline reproduction experiments
- `step-2-*.md` - Teacherless and coding rate experiments
- `step-3-crate-*.md` - CRATE integration experiments
- `archive/` - Superseded experiments

## Index

### Step 1: Baseline Reproduction

| Experiment | Date | Status | Key Finding |
|------------|------|--------|-------------|
| [baseline-smoke](step-1-baseline-smoke.md) | 2026-01-03 | ✅ | Smoke test passes |
| [baseline-cifar100-paper](step-1-baseline-cifar100-paper.md) | 2026-01-05 | ✅ | **57.84% linacc, 0.739 R²** - Paper-level |

### Step 2: Teacherless + Rate

| Experiment | Date | Status | Key Finding |
|------------|------|--------|-------------|
| [exp-a-ema-rate](step-2-exp-a-ema-rate.md) | 2026-01-05 | ✅ | EMA+Rate works: 34.1% linacc, 0.30 R² |
| [exp-b-stopgrad](step-2-exp-b-stopgrad.md) | 2026-01-05 | ✅ | Needs warmup+cosine: 30.0% linacc |
| [exp-c-symmetric](step-2-exp-c-symmetric.md) | 2026-01-05 | ✅ | No clear benefit vs baseline |
| [exp-d-sharpening](step-2-exp-d-sharpening.md) | 2026-01-05 | ❌ | Sharpening hurts with cosine loss |
| [mnist-subspace](step-2-mnist-subspace.md) | 2026-01-05 | ✅ | Subspace diagnostics validated |
| [remote-baselines](step-2-remote-baselines.md) | 2026-01-05 | 🔄 | Remote comparison running |

## Archive

Superseded experiments moved to `archive/` with notes on what replaced them.

| Experiment | Date | Reason Archived |
|------------|------|-----------------|
| [baseline-cifar100-vast-10ep](archive/step-1-baseline-cifar100-vast-10ep.md) | 2026-01-04 | Replaced by 2000-epoch paper run |
