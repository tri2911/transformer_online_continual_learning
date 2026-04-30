# Public Release Checklist (`tri2911`)

Use this once to publish this folder as a public GitHub repo.

## 1) Verify local tree

```bash
cd /mnt/work/trn/works/research_paper/AIO/continuous_learning
find . -name __pycache__ -type d -exec rm -rf {} +
find . -name .pytest_cache -type d -exec rm -rf {} +
```

## 2) Initialize git (if not already)

```bash
git init
git branch -M main
```

## 3) Commit clean source only

```bash
git add .
git status
git commit -m "Initial public release: continual learning CIFAR benchmark"
```

## 4) Create public repo under `tri2911`

With GitHub CLI:

```bash
gh repo create tri2911/transformer_online_continual_learning --public --source=. --remote=origin --push
```

If repo already exists:

```bash
git remote add origin git@github.com:tri2911/transformer_online_continual_learning.git
git push -u origin main
```

## 5) Sanity checks after push

- Open the repo page and confirm:
  - No `outputs/`, `checkpoints/`, or raw CIFAR files tracked.
  - `README.md` and `docs/RUNBOOK.md` render correctly.
  - `src/`, `scripts/`, `tests/` are present.
