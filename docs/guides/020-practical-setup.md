# Practical setup (local + remote)

## Goals
- Smoke tests on laptop (few batches, no crashes, loss decreases).
- Tiny but meaningful benchmark that detects collapse and equivariant vs invariant leakage.
- One logging tool for local + remote (W&B).

## 1) Local machine setup (Mac M3 Max)

### 1.1 Python environment (venv)
```bash
cd seq-jepa
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

If `requirements.txt` fails on macOS (e.g., `deepgaze-pytorch` not available), use:
```bash
pip install -r requirements-local.txt
```

### 1.2 Device auto-select (CUDA, MPS, CPU)
Use once in the wrapper training script and reuse everywhere:
```python
import torch

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

### 1.3 MPS fallback for unsupported ops (optional)
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### 1.4 DataLoader defaults for laptop
- Start with `num_workers=0`.
- Increase to 2-4 only if stable.
- Keep batch sizes small to avoid memory spikes.

## 2) Fast iteration suite (tiny but informative)

### 2.1 CIFAR-10 with discrete rotation actions
- Views are rotations in {0, 90, 180, 270} degrees.
- Sequence length M+1 = 4 views (three actions) in current configs.
- Action encoding uses sin/cos for rotation angles (continuous).

Why it works:
- Encoder output should retain pose (equivariance proxy).
- Aggregated state should discard pose but keep class (invariance proxy).
- If both reps perform equally well on both tasks, leakage is happening.

### 2.2 Subset support
- `subset_train` and `subset_val` select fixed random indices (seeded).
- This keeps runs short without changing dataset logic.

### 2.3 Standard run modes
Mode A - Smoke
- subset_train=128, max_steps=20, eval=off

Mode B - Quick
- subset_train=5000, subset_val=1000, short epochs

Mode C - Baseline-ish
- larger subset or full CIFAR-10

CLI pattern:
```bash
python train.py --config configs/quick/cifar10_rot_smoke.yaml
python train.py --config configs/quick/cifar10_rot_quick.yaml
python train.py --config configs/quick/cifar10_rot_baseline.yaml
```

### 2.4 Local gates (must pass before remote runs)
- Smoke: completes without crash, loss finite, `online_linacc_test` above random.
- Quick: `online_linacc_test` and `online_r2_test` exceed thresholds in config.
- Gate status is logged to W&B as `gate_pass` with `gate_reasons` in the run summary.
- See `docs/reference/metrics-and-gates.md` for full metric definitions and gates.

## 3) Logging and remote runs

### 3.1 W&B
- Set `WANDB_API_KEY` in the environment or run `wandb login` once.
- Use one project name for local + remote runs.
- For local smoke tests, use `WANDB_MODE=offline` if desired.
- Prefer `.env` for local credentials; use `.env.example` as the template.
- `train.py` loads `.env` automatically if present.

### 3.2 Vast.ai
- Use the same code and configs as local.
- Mount the repo and set `WANDB_API_KEY` in the container.
- Run the same `train.py --config ...` commands to keep results comparable.

## Known upstream issues
- `seq-jepa/src/engine.py` and `seq-jepa/src/models.py` currently contain indentation errors.
- `train.py` falls back to `experiments/models_seqjepa.py` to allow local smoke tests without editing upstream files.
