from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import gc
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from continuous_learning.config import ModelConfig, StreamConfig, TrainingConfig
from continuous_learning.data import load_cifar100
from continuous_learning.training import run_online_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGG++ joint-training suite (separate from pretrained runs).")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--final-eval-examples-per-task", type=int, default=500)
    parser.add_argument("--base-out", type=str, default="outputs/vgg_plus_plus")
    return parser.parse_args()


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _summary_from_run(method: str, result: dict[str, Any]) -> dict[str, float | str]:
    step_metrics = result.get("step_metrics", [])
    anchor_task_accuracy = result.get("anchor_task_accuracy", {})
    final_task_accuracy = result.get("final_task_accuracy", {})
    avg_all = _safe_mean([float(x.get("instant_accuracy", 0.0)) for x in step_metrics])
    last20 = []
    for task_id in range(80, 100):
        key = str(task_id)
        if key in anchor_task_accuracy:
            last20.append(float(anchor_task_accuracy[key]))
        elif task_id in anchor_task_accuracy:
            last20.append(float(anchor_task_accuracy[task_id]))
    avg_last20 = _safe_mean(last20)
    bwt_terms = []
    for task_id in range(0, 99):
        k = str(task_id)
        online = float(anchor_task_accuracy.get(k, anchor_task_accuracy.get(task_id, 0.0)))
        final = float(final_task_accuracy.get(k, final_task_accuracy.get(task_id, 0.0)))
        if online > 0.0 or final > 0.0:
            bwt_terms.append(final - online)
    return {
        "method": method,
        "avg_acc_all_tasks": avg_all,
        "avg_acc_last_20_tasks": avg_last20,
        "bwt": _safe_mean(bwt_terms),
    }


def _requested_payload(
    *,
    method: str,
    seed: int,
    stream_cfg: StreamConfig,
    model_cfg: ModelConfig,
    training_cfg: TrainingConfig,
    result: dict[str, Any],
) -> dict[str, Any]:
    step_metrics = result.get("step_metrics", [])
    anchor_task_accuracy = result.get("anchor_task_accuracy", {})
    final_task_accuracy = result.get("final_task_accuracy", {})
    summary = _summary_from_run(method, result)
    task_ids = sorted({int(k) for k in set(anchor_task_accuracy.keys()) | set(final_task_accuracy.keys())})
    return {
        "method": method,
        "feature_extractor": "vgg_plus_plus",
        "jointly_trained": True,
        "seed": seed,
        "config": {
            "stream": asdict(stream_cfg),
            "model": asdict(model_cfg),
            "training": asdict(training_cfg),
        },
        "per_step": {
            "t": [int(x["global_examples"]) for x in step_metrics],
            "acc": [float(x["instant_accuracy"]) for x in step_metrics],
            "loss": [float(x["loss"]) for x in step_metrics],
            "task_id": [int(x["task_id"]) for x in step_metrics],
        },
        "per_task": {
            "task_id": task_ids,
            "mean_acc": [float(anchor_task_accuracy.get(str(t), anchor_task_accuracy.get(t, 0.0))) for t in task_ids],
            "final_acc": [float(final_task_accuracy.get(str(t), final_task_accuracy.get(t, 0.0))) for t in task_ids],
        },
        "summary": {
            "avg_acc": float(summary["avg_acc_all_tasks"]),
            "last20_acc": float(summary["avg_acc_last_20_tasks"]),
            "bwt": float(summary["bwt"]),
        },
    }


def _load_pretrained_avg_acc(base_out: Path) -> dict[str, float]:
    pre_csv = Path("outputs/pretrained/runs/comparison_seed_rows.csv")
    if not pre_csv.exists():
        pre_csv = Path("outputs/runs/pretrained/comparison_seed_rows.csv")
    if not pre_csv.exists():
        pre_csv = Path("outputs/runs/comparison_seed_rows.csv")
    if not pre_csv.exists():
        return {}
    out: dict[str, list[float]] = {}
    with pre_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = str(row.get("method", "")).strip()
            if not method:
                continue
            out.setdefault(method, []).append(float(row.get("avg_acc_all_tasks", 0.0)))
    return {k: _safe_mean(v) for k, v in out.items()}


def _build_pretrained_vs_vggpp_figure(*, base_out: Path, vgg_summary: dict[str, float]) -> None:
    pre = _load_pretrained_avg_acc(base_out)
    if not pre:
        print("[warn] pretrained summary not found; skipping pretrained_vs_vggpp_comparison.png")
        return

    methods = ["online_sgd", "experience_replay", "pi_transformer", "two_token_transformer"]
    x = np.arange(len(methods), dtype=np.float32)
    pre_vals = [float(pre.get(m, 0.0)) for m in methods]
    vgg_vals = [float(vgg_summary.get(m, 0.0)) for m in methods]
    w = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar(x - w / 2, pre_vals, width=w, color="#1f77b4", label="pretrained")
    plt.bar(x + w / 2, vgg_vals, width=w, color="#ff7f0e", label="vgg_plus_plus joint")
    plt.xticks(x, methods, rotation=20)
    plt.ylabel("avg_acc")
    plt.title("Pretrained ResNet-50 vs Joint VGG++ Training")
    plt.legend()
    plt.tight_layout()
    out_path = Path("outputs/comparison/pretrained_vs_vggpp_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _cleanup_artifacts(*, runs_dir: Path, checkpoint_root: Path) -> None:
    for pyc_dir in runs_dir.rglob("__pycache__"):
        if pyc_dir.is_dir():
            for f in pyc_dir.glob("*"):
                f.unlink(missing_ok=True)
            pyc_dir.rmdir()

    one_gb = 1024**3
    for pt_file in checkpoint_root.rglob("*.pt"):
        if pt_file.name in {"latest.pt", "best.pt"}:
            continue
        if pt_file.stat().st_size > one_gb:
            pt_file.unlink(missing_ok=True)


def _cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _gpu_process_snapshot() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        return out
    except Exception:
        return "nvidia-smi unavailable"


def main() -> None:
    args = parse_args()
    base_out = Path(args.base_out)
    runs_dir = base_out / "runs"
    raw_dir = runs_dir / "raw"
    figures_dir = base_out / "figures"
    report_dir = base_out / "report"
    checkpoint_root = Path("checkpoints") / "vgg_plus_plus"

    runs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    stream_cfg = StreamConfig()
    dataset = load_cifar100(root=args.data_root, config=stream_cfg, train=True, download=True)
    labels = np.asarray(dataset.targets if hasattr(dataset, "targets") else dataset.labels, dtype=np.int64)
    images_np = dataset.data

    method_specs = [
        ("pi_transformer", "pi", 20, 512),
        ("experience_replay", "two_token", 20, 1),
        ("two_token_transformer", "two_token", 20, 512),
        ("online_sgd", "online_sgd", 1, 1),
    ]

    summary_rows: list[dict[str, float | str]] = []
    vgg_avg: dict[str, float] = {}

    for method_id, model_kind, replay_streams, context_window in method_specs:
        _cuda_cleanup()
        model_cfg = ModelConfig(context_window=context_window)
        training_cfg = TrainingConfig(
            replay_streams=replay_streams,
            max_steps=args.max_steps,
            device=args.device,
            log_every=args.log_every,
            use_feature_cache=False,
        )
        ckpt_dir = checkpoint_root / f"{method_id}_{args.seed}"
        ckpt_path = ckpt_dir / "latest.pt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        print(f"[start] {method_id} seed={args.seed}")
        try:
            result = run_online_training(
                images_np=images_np,
                labels_np=labels,
                stream_config=stream_cfg,
                model_config=model_cfg,
                training_config=training_cfg,
                seed=args.seed,
                model_kind=model_kind,
                verbose=True,
                run_name=f"[vgg++/{method_id}]",
                collect_diagnostics=True,
                anchor_stream_no_reset=True,
                recovery_examples=200,
                compute_final_task_eval=True,
                final_eval_max_examples_per_task=args.final_eval_examples_per_task,
                feature_extractor_kind="vgg_plus_plus",
                checkpoint_path=str(ckpt_path),
                checkpoint_every=args.checkpoint_every,
                resume=True,
            )
        except torch.OutOfMemoryError as e:
            _cuda_cleanup()
            snapshot = _gpu_process_snapshot()
            raise RuntimeError(
                f"CUDA OOM during method='{method_id}'. "
                "This run resumes automatically from latest.pt, but GPU memory is currently insufficient.\n"
                f"GPU processes:\n{snapshot}\n"
                "Try rerunning with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                "and ensure enough free VRAM before retrying."
            ) from e

        summary = _summary_from_run(method_id, result)
        summary_rows.append(summary)
        vgg_avg[method_id] = float(summary["avg_acc_all_tasks"])

        payload = _requested_payload(
            method=method_id,
            seed=args.seed,
            stream_cfg=stream_cfg,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            result=result,
        )
        out_json = runs_dir / f"run_{method_id}_{args.seed}.json"
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        raw_payload = {
            "method": method_id,
            "feature_extractor": "vgg_plus_plus",
            "jointly_trained": True,
            "seed": args.seed,
            "stream_config": asdict(stream_cfg),
            "model_config": asdict(model_cfg),
            "training_config": asdict(training_cfg),
            "result": result,
        }
        raw_out = raw_dir / f"{method_id}_seed{args.seed}.json.gz"
        with gzip.open(raw_out, "wt", encoding="utf-8") as f:
            json.dump(raw_payload, f)
        print(f"[done] {method_id} -> {out_json}")
        _cuda_cleanup()

    index_path = runs_dir / "runs_index.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "method": str(r["method"]),
                    "seed": args.seed,
                    "path": str(raw_dir / f"{r['method']}_seed{args.seed}.json.gz"),
                    "summary": r,
                }
                for r in summary_rows
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    csv_path = runs_dir / "comparison_seed_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "avg_acc_all_tasks", "avg_acc_last_20_tasks", "bwt"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Build report figures for the new run.
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "build_report_figures.py"),
            "--runs-dir",
            str(runs_dir),
            "--out-dir",
            str(report_dir),
            "--figures-dir",
            str(figures_dir),
            "--ablation-results",
            "",
        ],
        check=True,
    )

    _build_pretrained_vs_vggpp_figure(base_out=base_out, vgg_summary=vgg_avg)
    _cleanup_artifacts(runs_dir=runs_dir, checkpoint_root=checkpoint_root)
    print(f"[saved] runs={runs_dir}")
    print(f"[saved] figures={figures_dir}")
    print(f"[saved] report={report_dir}")


if __name__ == "__main__":
    main()
