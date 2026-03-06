## DDPM Mode-Interpolation Reproduction (Toy)

This repo contains a **from-scratch toy implementation** for the class demo:

- Phase 1: DDPM on 1D / 2D Gaussian mixtures
  - ablation with sampling respacing (`1000`, `250`, `50`)
  - generated-vs-true density plots
  - bridge-mass measurement between modes
  - trajectory-variance metric `Hal(x)` from late `\hat{x}_0(t)` estimates
- Phase 2: 1D oracle-score baseline
  - replaces NN score with analytic score of `q_t`
  - shows bridge density collapse

## Quick start (Mac M4 Pro or Colab)

### 1) Install dependencies

```bash
pip install torch torchvision torchaudio matplotlib numpy tqdm
```

### 2) Run both phases

```bash
python experiments/ddpm_gaussian_dataset.py --phase both
```

Results are saved to:

- `results/toy_mode_interp/phase1_1d_ablation_density.png`
- `results/toy_mode_interp/phase1_1d_hal_hist.png`
- `results/toy_mode_interp/phase1_2d_ablation_scatter.png`
- `results/toy_mode_interp/phase2_1d_oracle_vs_nn.png`
- `results/toy_mode_interp/metrics_summary.json`

## Recommended runtime setup

- **Easiest for this toy project: your MacBook M4 Pro**
  - 1D/2D MLP DDPM is lightweight.
  - Script auto-selects `mps` when available (`--device auto`).
- **Use Colab only if you want faster iteration or shared notebooks.**
  - Same script runs unchanged.
  - Set `--device cuda` in Colab.

## Useful flags

```bash
# Faster debug run (quick sanity check)
python experiments/ddpm_gaussian_dataset.py \
  --phase both \
  --epochs_1d 40 --epochs_2d 60 --batch_size 1024 \
  --sample_count 50000 --sample_count_2d 40000

# Phase 1 ablation requested in class spec:
# same trained model, vary sampling respacing (1000/250/50)
python experiments/ddpm_gaussian_dataset.py \
  --phase phase1 \
  --ablation_target sampling_steps \
  --ablation_values 1000,250,50

# Only phase 2 oracle baseline
python experiments/ddpm_gaussian_dataset.py --phase phase2

# Alternative ablation:
# vary diffusion timestep hyperparameter T (retrain each model)
python experiments/ddpm_gaussian_dataset.py \
  --phase phase1 \
  --ablation_target diffusion_timesteps \
  --ablation_values 1000,250,50
```

Notes:
- `sample_count` and `sample_count_2d` control how many generated points are used to estimate densities/bridge mass.
- Ablation now explicitly separates:
  - `sampling_steps` (respacing at sampling time), and
  - `diffusion_timesteps` (the DDPM hyperparameter `T` used during training and full sampling).

## Your requested full comparison runs

Run both commands below to compare bridge density with full vs half training data, while keeping:
- fixed `T=1000` during training,
- sampling-step ablation `1000,250,50`,
- larger evaluation sample counts (x5 defaults).

```bash
# A) Full training size
python experiments/ddpm_gaussian_dataset.py \
  --phase phase1 \
  --T 1000 \
  --ablation_target sampling_steps \
  --ablation_values 1000,250,50 \
  --eta 1.0 \
  --train_size_1d 50000 \
  --train_size_2d 100000 \
  --sample_count 10000000 \
  --sample_count_2d 5000000 \
  --eval_batch_size_1d 100000 \
  --eval_batch_size_2d 100000 \
  --hal_eval_count 200000 \
  --hal_batch_size_1d 50000 \
  --outdir results/fixedT1000_full_train

# B) Half training size
python experiments/ddpm_gaussian_dataset.py \
  --phase phase1 \
  --T 1000 \
  --ablation_target sampling_steps \
  --ablation_values 1000,250,50 \
  --eta 1.0 \
  --train_size_1d 25000 \
  --train_size_2d 50000 \
  --sample_count 10000000 \
  --sample_count_2d 5000000 \
  --eval_batch_size_1d 100000 \
  --eval_batch_size_2d 100000 \
  --hal_eval_count 200000 \
  --hal_batch_size_1d 50000 \
  --outdir results/fixedT1000_half_train
```

Then compare:
- `results/fixedT1000_full_train/metrics_summary.json`
- `results/fixedT1000_half_train/metrics_summary.json`

Bridge-mass values for each sampling-step setting are in `phase1.ablation`.
Use `--eta 1.0` for stochastic DDPM-like sampling (recommended for mode-interpolation effects).