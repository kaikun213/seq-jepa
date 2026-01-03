# Experiment Logs

This directory tracks all experiments: what was tried, results, and decisions.

## Structure

- `experiments/` - Individual experiment logs
  - `step-1-baseline-*.md` - Baseline reproduction experiments
  - `step-2-teacherless-*.md` - Teacherless and coding rate experiments
  - `step-3-crate-*.md` - CRATE integration experiments
  - etc.

## Experiment Log Template

Each experiment should follow this structure:

```markdown
# Experiment: [Short Name]

**Date**: YYYY-MM-DD
**Step**: [Step number/name]
**Status**: ✅ Success | ❌ Failed | 🔄 In Progress | ⏸️ Paused

## Objective
[What we're trying to achieve]

## Setup
- Config: `configs/path/to/config.yaml`
- Model variant: [e.g., SeqJEPA_Transforms with EMA]
- Dataset: [e.g., CIFAR-10 rotations]
- Hyperparameters:
  - lr: 0.001
  - lambda_rate: 0.01
  - etc.

## Results
- Metrics: [Key numbers]
- W&B run: [link or run ID]
- Observations: [What happened]

## Analysis
- What worked: [Success factors]
- What didn't: [Failure modes]
- Surprises: [Unexpected findings]

## Decisions
- [Decision made based on this experiment]
- [Next steps or follow-ups]

## Files Changed
- `experiments/models_seqjepa.py`: [what changed]
- `train.py`: [what changed]
- etc.
```

## Index

| Experiment | Step | Date | Status | Key Finding |
|------------|------|------|--------|-------------|
| baseline-smoke | Step 1 | - | ✅ | Smoke test passes |
| baseline-quick | Step 1 | - | 🔄 | In progress |

