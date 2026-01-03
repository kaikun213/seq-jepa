# Step 0 - Repo setup and diagnostics

## Goal
- Reproducible environment and a reliable view of where to edit.
- Standardized logging and collapse diagnostics before any architecture changes.

## Tasks
- Create a clean venv and install requirements.
- Locate entrypoints and core modules.
- Add or verify logging, checkpoints, and run metadata.
- Add collapse + structure diagnostics for z_t and z_AGG.

## Commands to locate code
```bash
rg -n "if __name__ == .__main__." -S seq-jepa
rg -n "argparse|hydra|omegaconf|click" -S seq-jepa
find seq-jepa -maxdepth 3 -type f -name "*.py"

rg -n "AGG|\\[AGG\\]|CLS" -S seq-jepa
rg -n "EMA|moving average|momentum encoder|target encoder" -S seq-jepa
rg -n "cosine|CosineSimilarity|F\\.cosine_similarity" -S seq-jepa
rg -n "action|a_M|relative|delta|transform" -S seq-jepa
rg -n "torch\\.cat\\(|concat" -S seq-jepa
rg -n "linear probe|linear_probe|eval|evaluation|downstream" -S seq-jepa
```

## Diagnostics to log each batch
- mean/std per feature dim for z_t and z_AGG
- effective rank or eigenvalue spectrum of batch covariance
- mean cosine similarity across samples
- sparsity stats later when CRATE/ISTA is added

## Deliverables
- Reproducible environment instructions.
- Verified entrypoints and key modules.
- Collapse diagnostics visible in logs.
