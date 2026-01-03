# Remote run setup (Vast.ai + SLURM)

## Goal
- Run Step 1 baseline reproduction with minimal overhead using the lightest wired dataset.
- Keep configs consistent across local and remote runs.

## Dataset note
- The wrapper runner supports `dataset.name=cifar100_aug` (paper-aligned transforms) and `dataset.name=cifar10_rot` (fast sanity).
- Step 1 baseline uses CIFAR-100 augmentation actions; CIFAR-10 rotations stay as a quick fallback.

## Recommended configs
- Baseline (CIFAR-100 aug): `configs/remote/cifar100_aug_baseline.yaml`
- Shakedown (CIFAR-100 aug): `configs/quick/cifar100_aug_smoke.yaml` or `configs/quick/cifar100_aug_quick.yaml`
- Optional fast fallback: `configs/quick/cifar10_rot_smoke.yaml` or `configs/quick/cifar10_rot_quick.yaml`

## Common prerequisites
- W&B API key: set `WANDB_API_KEY` or place it in `.env` (loaded by `train.py`).
- Python env with `requirements.txt` installed.
- CIFAR-100 will download into `data/` on first run. If compute nodes have no internet, pre-populate `data/` before submission.
  
## Private repo access (Vast.ai)
- Preferred: add your GitHub SSH key to the instance (`~/.ssh/id_ed25519`) and clone via SSH.
- HTTPS fallback: set `GITHUB_TOKEN` and clone with `https://github.com/kaikun213/seq-jepa-streaming.git`.
- If using the Vast.ai CLI, set `VAST_API_KEY`.

## Vast.ai
- Suggested instance: 1x GPU (RTX 3060 12GB or A10 24GB), 8 vCPU, 30-60GB disk.
- Run:
  - `git clone` the repo
  - `pip install -r requirements.txt`
  - `scripts/vast/run_cifar100_aug_baseline.sh`
- Optional: run the CIFAR-100 smoke/quick config first to validate the environment.

## SLURM
- Template: `scripts/slurm/seqjepa_cifar10_rot_baseline.sbatch` (update to CIFAR-100 when needed)
- Fill in partition/account, module loads, and env activation.
- Submit with `sbatch scripts/slurm/seqjepa_cifar10_rot_baseline.sbatch`

## Rough runtimes (very approximate)
- Mac M3 Max: quick config ~8 min (based on recent 5-10 min runs).
- Vast.ai 1x RTX 3060: quick ~1-2 min, baseline ~4-6 min.
- First run adds data download time (~1-3 min) if not cached.

## If you hit issues
- CUDA OOM: lower `dataset.batch_size` to 64.
- Slow dataloading: increase `dataset.num_workers` to 4-8 on GPU nodes.

## SSL Certificate Setup for Zscaler

If you're behind a Zscaler proxy, you need to configure SSL certificates for Python/conda to work with vastai and other tools.

### Quick Setup

1. **Activate the Python env you will run `vastai` from**, then add the Zscaler certificate to that env's certifi bundle:
   ```bash
   ./fix_ssl_certs.sh
   ```

2. **The script automatically:**
   - Backs up your current certifi bundle
   - Adds Zscaler root certificate to the bundle
   - Shows you the certificate path

3. **Environment variables are set in `~/.zshrc`:**
   - `SSL_CERT_FILE` - Points to certifi bundle with Zscaler cert
   - `REQUESTS_CA_BUNDLE` - Same as SSL_CERT_FILE for requests library

### Python 3.13 SSL Strictness Issue

**Note:** Python 3.13 has stricter SSL certificate validation. If you see errors like:

`Basic Constraints of CA cert not marked critical`

then this is not fixable by just appending the cert. Use Python 3.11 or 3.12, or ask IT for a reissued Zscaler root CA that marks Basic Constraints as critical.

1. **Use Python 3.11 or 3.12 instead:**
   ```bash
   conda create -n py311 python=3.11
   conda activate py311
   pip install vastai
   ```

2. **Or configure network to bypass Zscaler** for specific domains (vast.ai, conda-forge, etc.)

3. **Or use the workaround** (less secure, only for testing):
   ```bash
   export PYTHONHTTPSVERIFY=0  # Not recommended for production
   ```

### Testing SSL Configuration

```bash
# Test basic SSL
python3 -c "import requests; print(requests.get('https://www.google.com', timeout=5).status_code)"

# Test vastai
vastai --help
```

### Files Created

- `fix_ssl_certs.sh` - Script to add Zscaler cert to certifi bundle
- `~/Documents/cacert/` - Directory with Zscaler certificates
- `~/.zshrc` - Updated with SSL environment variables
