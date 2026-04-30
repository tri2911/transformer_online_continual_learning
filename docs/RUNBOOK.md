# Continuous Learning Runbook

This repo now separates outputs into two experiment tracks:

- `outputs/pretrained/*`: frozen/pre-extracted feature track ("normal" run)
- `outputs/vgg_plus_plus/*`: joint VGG++ training track

Also:
- `outputs/comparison/pretrained_vs_vggpp_comparison.png`: direct comparison chart
- `outputs/logs/*`: run logs
- `checkpoints/vgg_plus_plus/*`: VGG++ resume checkpoints

## 1) Setup

```bash
python -m pip install -U pip
python -m pip install torch torchvision matplotlib pytest
```


```bash
pytest -q
```

## 2) Data

CIFAR-100 is downloaded automatically by training scripts into `data/`.

For pretrained/frozen-feature runs, pre-extract features once (optional but recommended):

```bash
PYTHONPATH=src python -m continuous_learning.data.extract_features \
  --checkpoint checkpoints/vggpp_cifar100.pt \
  --output data/features_cache.pt \
  --device cuda
```

## 3) Run Pretrained/Frozen Track

### 3.1 Main suite

```bash
python scripts/run_report_suite.py \
  --device cuda \
  --max-steps 10000 \
  --log-every 25 \
  --final-eval-examples-per-task 500
```

Writes to:
- `outputs/pretrained/runs/raw/*.json.gz`
- `outputs/pretrained/runs/comparison_seed_rows.csv`

### 3.2 Build report figures/tables

```bash
python scripts/build_report_figures.py \
  --runs-dir outputs/pretrained/runs \
  --out-dir outputs/pretrained/report \
  --figures-dir outputs/pretrained/figures \
  --ablation-results outputs/pretrained/ablation_results.json
```

### 3.3 Optional ablations + claim plots

```bash
python scripts/run_ablations.py --device cuda --max-steps 10000
python scripts/plot_claims.py --results outputs/pretrained/ablation_results.json
```

Writes to:
- `outputs/pretrained/ablation_results.json`
- `outputs/pretrained/figures/*.png`

## 4) Run Joint VGG++ Track

This suite includes:
- `pi_transformer`
- `experience_replay`
- `two_token_transformer`
- `online_sgd`

with joint VGG++ feature learning and checkpoint resume.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
python scripts/run_vggpp_joint_suite.py \
  --device cuda \
  --max-steps 10000 \
  --seed 42 \
  --log-every 25 \
  --checkpoint-every 500 \
  --final-eval-examples-per-task 500
```

Writes to:
- `outputs/vgg_plus_plus/runs/run_*_42.json`
- `outputs/vgg_plus_plus/runs/raw/*.json.gz`
- `outputs/vgg_plus_plus/report/*.csv|*.md`
- `outputs/vgg_plus_plus/figures/*.png`
- `checkpoints/vgg_plus_plus/<method>_42/{latest.pt,best.pt}`

### 4.1 Generate full claim set for VGG++ (same claims as pretrained)

Run VGG++ ablations + claim plots into VGG++ folders:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
python scripts/run_ablations.py \
  --device cuda \
  --max-steps 10000 \
  --no-feature-cache \
  --feature-extractor-kind vgg_plus_plus \
  --out outputs/vgg_plus_plus/ablation_results.json

python scripts/plot_claims.py \
  --results outputs/vgg_plus_plus/ablation_results.json \
  --out-dir outputs/vgg_plus_plus/figures

python scripts/build_report_figures.py \
  --runs-dir outputs/vgg_plus_plus/runs \
  --out-dir outputs/vgg_plus_plus/report \
  --figures-dir outputs/vgg_plus_plus/figures \
  --ablation-results outputs/vgg_plus_plus/ablation_results.json
```

## 5) Compare Pretrained vs VGG++

Automatically produced by VGG++ runner:

- `outputs/comparison/pretrained_vs_vggpp_comparison.png`

## 6) Resume Behavior

- Pretrained/frozen scripts can be rerun safely (they overwrite run artifacts).
- VGG++ joint suite resumes from:
  - `checkpoints/vgg_plus_plus/<method>_42/latest.pt`

Run the same command again to resume after interruption.
