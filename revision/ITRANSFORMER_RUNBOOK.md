# iTransformer baseline — cloud runbook (deadline: 12-06)

Modern-DL baseline (iTransformer, Liu et al., ICLR 2024) for **Protocol B**, built to
drop straight into Table 4 / Figs 2,4,5. Same data pipeline, feature set, per-station
scaling, sequence-eligibility, and metrics as `lstm_global_attention.py` — only the
model differs (self-attention across **variate tokens** + a learned **station token**,
residual-anchored to last observed PM2.5).

Script: `train_models/global/itransformer_global.py`

## STEP 0 — Machine setup & verification (run on EVERY box first)

> Vast.ai instances are Docker containers and may drop your SSH session. Do everything
> inside `tmux` so a disconnect never kills a multi-hour run.

```bash
# 0.1 Persistent session (reconnect later with: tmux attach -t it)
tmux new -s it

# 0.2 Confirm the GPU + driver are actually present (driver must be >=535 for TF 2.20)
nvidia-smi          # should list the RTX 4090/3090 and a CUDA >=12.x driver

# 0.3 Get the code (clone your repo, or scp the single script + scripts/ dir)
git clone <YOUR_REPO_URL> eea-pm25 && cd eea-pm25
#   (or: copy train_models/global/itransformer_global.py into place and cd to repo root)

# 0.4 Minimal, fast install (iTransformer only needs these — skip the heavy ML/DL extras)
pip install --upgrade pip
pip install "tensorflow[and-cuda]==2.20.*" pandas numpy pyarrow
#   tensorflow[and-cuda] bundles the matching CUDA 12.x libs, so only the host DRIVER matters.

# 0.5 Pull the dataset from Hugging Face into the repo ROOT (adjust repo id / filename)
pip install -U "huggingface_hub[cli]"
hf download cosuleabianca/eea-pm25-dataset ml_ready_dataset_full_realistic.csv \
    --repo-type dataset --local-dir .
ls -lh ml_ready_dataset_full_realistic.csv     # must be in the repo root (~1-2 GB)
```

### 0.6 GPU sanity check — DO NOT train until this passes
```bash
python -c "import tensorflow as tf; \
print('TF', tf.__version__); \
gpus=tf.config.list_physical_devices('GPU'); \
print('GPUs:', gpus); \
assert gpus, 'NO GPU VISIBLE — fix CUDA/driver before training (else it silently runs on CPU)'; \
import numpy as np; \
x=tf.random.normal((1024,1024)); _=tf.matmul(x,x); \
print('GPU matmul OK')"
```
Expected: `TF 2.20.x`, a non-empty `GPUs:` list, and `GPU matmul OK`. If `GPUs:` is empty,
TF will train on CPU (much slower) — stop and fix the image/driver first.

> Always run the training **from the repo root** (paths in the script are relative).

## STEP 1 — Canary (do this FIRST, one machine)
Train only h=1 and sanity-check before committing all machines:

```bash
python train_models/global/itransformer_global.py --horizon 1
```

**Go/no-go check** — open `results/itransformer_metrics_h1.csv`, `Scope=Global` row:
- RMSE should land roughly **2.1–2.6** (cf. Table 4: LSTM-Attn 2.18, LGBM 2.23, persistence 2.42).
- R2 ~0.90, Bias near 0 (|Bias| < ~0.3).

If RMSE is wildly off (e.g. >4, or NaN, or ≫ persistence) → STOP, ping me; likely a
normalization/inverse-transform issue. Do **not** launch the rest.

Canary wall-clock: h=1 early-stops fast (the LSTMs hit ~7 epochs at h=1) — expect ~1–3 h.

## STEP 2 — Launch remaining horizons, one per machine (parallel)
```bash
python train_models/global/itransformer_global.py --horizon 3     # machine 2
python train_models/global/itransformer_global.py --horizon 6     # machine 3
python train_models/global/itransformer_global.py --horizon 12    # machine 4
python train_models/global/itransformer_global.py --horizon 24    # machine 5
```
Wall-clock ≈ the single slowest horizon (h12/h24), ballpark **4–7 h**.

**Time-pressure lever:** cap long horizons to bound worst case, e.g.
`--epochs 30` for h12/h24 (early stopping usually fires earlier anyway).

## STEP 3 — Collect + merge (one machine)
Copy back these per-horizon artifacts from each machine into the repo:
- `results/itransformer_metrics_h{H}.csv`
- `results/itransformer_pred_sample_h{H}.csv`
- `train_models/global/runs_info/itransformer_h{H}_run_info.json`
- `models/itransformer_models/itransformer_h{H}.keras` (optional, for HF upload)

Then:
```bash
python train_models/global/itransformer_global.py --merge
```
Produces the combined, paper-ready files:
- `results/itransformer_metrics.csv`  (Scope/Group/horizon/MAE/RMSE/R2/Bias/Count — same schema as every other model)
- `results/itransformer_predictions_sample.csv`
- `train_models/global/runs_info/itransformer_run_info.json`

## STEP 4 — Fold into paper outputs
- `python scripts/compute_model_skill.py` (adds iTransformer to `results_with_skill/`)
- `python scripts/generate_publication_outputs.py` (Table 4 + Figs 2/4/5 pick it up)
- Optional: add iTransformer to the DM test pair list and the SHAP/figure regen.

## Single-machine fallback (no parallelism)
```bash
python train_models/global/itransformer_global.py        # all 5 horizons, ~13–24 h, auto-merges
```

## Notes
- One model per horizon, direct single-output (matches the rest of the benchmark).
- Architecture knobs at top of the script: `D_MODEL=128, N_HEADS=8, FF_DIM=256,
  N_BLOCKS=2, DROPOUT=0.2`. No HPO (baseline) — single sane config, fixed seed 42.
- Each `--horizon` run writes its own scalers file (`itransformer_scalers_h{H}.json`),
  so parallel machines never collide.
