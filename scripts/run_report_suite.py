from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from continuous_learning.config import ModelConfig, StreamConfig, TrainingConfig
from continuous_learning.data import load_cifar100
from continuous_learning.training import run_online_training


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    model_kind: str
    replay_streams: int
    context_window: int
    description: str


METHOD_SPECS: list[MethodSpec] = [
    MethodSpec(
        method_id="online_sgd",
        model_kind="two_token",
        replay_streams=1,
        context_window=1,
        description="No replay, no long context baseline.",
    ),
    MethodSpec(
        method_id="experience_replay",
        model_kind="two_token",
        replay_streams=20,
        context_window=1,
        description="Replay baseline without long-context temporal modeling.",
    ),
    MethodSpec(
        method_id="two_token_transformer",
        model_kind="two_token",
        replay_streams=20,
        context_window=512,
        description="Two-token temporal baseline.",
    ),
    MethodSpec(
        method_id="pi_transformer",
        model_kind="pi",
        replay_streams=20,
        context_window=512,
        description="Full Pi-Transformer.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full report experiment suite.")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-steps", type=int, default=10_000, help="10,000 * S=50 -> 500,000 examples.")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed-main", type=int, default=42)
    parser.add_argument("--seeds-variance", type=str, default="42,123,456")
    parser.add_argument("--variance-methods", type=str, default="pi_transformer,two_token_transformer")
    parser.add_argument(
        "--final-eval-examples-per-task",
        type=int,
        default=500,
        help="Use 5000 for full task-size evaluation; lower values are faster.",
    )
    parser.add_argument("--features-cache", type=str, default="data/features_cache.pt")
    parser.add_argument("--use-feature-cache", action="store_true", default=True)
    parser.add_argument("--no-feature-cache", action="store_false", dest="use_feature_cache")
    parser.add_argument("--out-dir", type=str, default="outputs/pretrained/runs")
    return parser.parse_args()


def _parse_seed_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _summary_row(method_id: str, seed: int, result: dict[str, object]) -> dict[str, object]:
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
        online = (
            float(anchor_task_accuracy[k])
            if k in anchor_task_accuracy
            else float(anchor_task_accuracy.get(task_id, 0.0))
        )
        final = (
            float(final_task_accuracy[k])
            if k in final_task_accuracy
            else float(final_task_accuracy.get(task_id, 0.0))
        )
        if online > 0.0 or final > 0.0:
            bwt_terms.append(final - online)
    bwt = _safe_mean(bwt_terms)

    return {
        "method": method_id,
        "seed": seed,
        "avg_acc_all_tasks": avg_all,
        "avg_acc_last_20_tasks": avg_last20,
        "bwt": bwt,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    def log(msg: str) -> None:
        elapsed = time.perf_counter() - start
        print(f"[{elapsed:8.1f}s] {msg}", flush=True)

    stream_cfg = StreamConfig()
    images_np: np.ndarray | None = None
    feature_bank: torch.Tensor | None = None
    labels: np.ndarray
    if args.use_feature_cache and Path(args.features_cache).exists():
        log(f"Loading feature cache labels from {args.features_cache}...")
        cache = torch.load(args.features_cache, map_location="cpu")
        labels = cache["labels"].cpu().numpy().astype(np.int64)
        feature_bank = cache["features"].contiguous()
        log(f"Feature cache ready: {tuple(feature_bank.shape)}")
    else:
        log("Loading CIFAR-100 once for all report runs...")
        dataset = load_cifar100(root=args.data_root, config=stream_cfg, train=True, download=True)
        labels = np.asarray(dataset.targets if hasattr(dataset, "targets") else dataset.labels, dtype=np.int64)
        images_np = dataset.data
        log(f"Dataset ready: {len(labels)} examples.")

    variance_methods = {x.strip() for x in args.variance_methods.split(",") if x.strip()}
    variance_seeds = _parse_seed_list(args.seeds_variance)

    summary_rows: list[dict[str, object]] = []
    run_index: list[dict[str, object]] = []
    total_jobs = 0
    for spec in METHOD_SPECS:
        seeds = variance_seeds if spec.method_id in variance_methods else [args.seed_main]
        total_jobs += len(seeds)

    job = 0
    for spec in METHOD_SPECS:
        seeds = variance_seeds if spec.method_id in variance_methods else [args.seed_main]
        for seed in seeds:
            job += 1
            log(f"[{job}/{total_jobs}] {spec.method_id} seed={seed} (start)")
            model_cfg = ModelConfig(context_window=spec.context_window)
            training_cfg = TrainingConfig(
                replay_streams=spec.replay_streams,
                max_steps=args.max_steps,
                device=args.device,
                log_every=args.log_every,
                features_cache=args.features_cache,
                use_feature_cache=args.use_feature_cache,
            )
            run = run_online_training(
                images_np=images_np,
                labels_np=labels,
                feature_bank=feature_bank,
                stream_config=stream_cfg,
                model_config=model_cfg,
                training_config=training_cfg,
                seed=seed,
                model_kind=spec.model_kind,
                verbose=True,
                run_name=f"[{spec.method_id}]",
                collect_diagnostics=True,
                anchor_stream_no_reset=True,
                recovery_examples=200,
                compute_final_task_eval=True,
                final_eval_max_examples_per_task=args.final_eval_examples_per_task,
            )

            payload = {
                "method": spec.method_id,
                "description": spec.description,
                "seed": seed,
                "stream_config": asdict(stream_cfg),
                "model_config": asdict(model_cfg),
                "training_config": asdict(training_cfg),
                "result": run,
            }
            out_file = raw_dir / f"{spec.method_id}_seed{seed}.json.gz"
            with gzip.open(out_file, "wt", encoding="utf-8") as f:
                json.dump(payload, f)

            summary = _summary_row(spec.method_id, seed, run)
            summary_rows.append(summary)
            run_index.append(
                {
                    "method": spec.method_id,
                    "seed": seed,
                    "path": str(out_file),
                    "summary": summary,
                }
            )
            log(f"[{job}/{total_jobs}] {spec.method_id} seed={seed} (done) -> {out_file.name}")

    index_path = out_dir / "runs_index.json"
    index_path.write_text(json.dumps(run_index, indent=2), encoding="utf-8")

    csv_path = out_dir / "comparison_seed_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "seed", "avg_acc_all_tasks", "avg_acc_last_20_tasks", "bwt"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    log(f"Saved run index: {index_path}")
    log(f"Saved per-seed summary: {csv_path}")


if __name__ == "__main__":
    main()
