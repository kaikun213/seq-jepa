# Experiment Template

> **Usage**: Copy this template to `docs/experiments/` and rename to `step-N-experiment-name.md`.
> Delete this usage note and fill in the sections below.

---

# Experiment: [Descriptive Name]

**Date**: YYYY-MM-DD  
**Step**: Step N - [Step Name]  
**Status**: 🔄 In Progress | ✅ Complete | ❌ Failed | ⏸️ Paused

## Links

| Resource | Link |
|----------|------|
| W&B Run | [run-id](https://wandb.ai/...) |
| Config | `configs/path/to/config.yaml` |
| Vast Instance | ID: NNNNNN (if remote) |

## Objective

_What are we trying to learn or validate? State the hypothesis if applicable._

Example:
> Validate that adding coding-rate regularizer to EMA-based seq-JEPA maintains baseline performance while preventing collapse.

## Setup

### Configuration
- **Model**: [architecture details]
- **Dataset**: [name, size, seq_len]
- **Key hyperparameters**: [lr, batch_size, epochs, etc.]

### Hardware
- **Device**: [local MPS / Vast RTX 3090 / etc.]
- **Estimated runtime**: [hours]
- **Estimated cost**: [$X.XX if remote]

### Reproduction
```bash
# Command to reproduce
python train.py --config configs/path/to/config.yaml
```

## Results

### Summary

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| `online_linacc_test` | XX.X% | ≥N% | ✅/❌ |
| `online_r2_test` | X.XXX | ≥N | ✅/❌ |
| `leakage_linacc_test` | XX.X% | — | — |
| `leakage_r2_test` | X.XXX | — | — |
| `ep_loss` | X.XXXX | — | — |

### Detailed Results

_Include sub-experiments, ablations, or iterations if applicable._

#### Variant A: [Description]
- **Metrics**: ...
- **W&B**: [link]

#### Variant B: [Description]
- **Metrics**: ...
- **W&B**: [link]

## Observations

_What did we observe during training and evaluation? Include both expected and unexpected findings._

1. **[Finding 1]**: Description
2. **[Finding 2]**: Description
3. **[Finding 3]**: Description

## Preliminary Analysis

_Put results in context. This section is required for all experiments._

### Comparison to Reference Paper

_How do these results compare to the seq-JEPA paper or other baselines?_

- Paper reports: [X]
- We achieved: [Y]
- Delta: [interpretation]

### Implications for Next Steps

_What does this mean for the immediate next experiments?_

1. [Implication 1]
2. [Implication 2]

### Meaning for Larger Vision

_How does this connect to the broader project goals (streaming, continual learning, local computation)?_

- [Connection to M3-M6 milestones]
- [Impact on architecture decisions]

## Decisions

_What decisions were made based on these results?_

- [ ] [Decision 1]
- [ ] [Decision 2]

## Next Steps

_Concrete actions to take after this experiment._

- [ ] [Action 1]
- [ ] [Action 2]

## Files Changed

_List any code or config changes made for this experiment._

- `path/to/file.py`: [description of change]
- `configs/path/to/config.yaml`: [description]

---

## Appendix (Optional)

### Training Curves

_Include or link to relevant plots._

### Failed Attempts

_Document what didn't work and why (valuable for future reference)._

### Raw Logs

_Link to full logs if needed for debugging._

