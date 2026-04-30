from __future__ import annotations

import argparse
import json
from pathlib import Path
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from continuous_learning.config import ModelConfig, StreamConfig, TrainingConfig
from continuous_learning.data import load_cifar100
from continuous_learning.training import run_online_training
from continuous_learning.training.metrics import mean_recovery_window_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run core ablations from to_do.md")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="outputs/pretrained/ablation_results.json")
    parser.add_argument("--freeze-at-step", type=int, default=5_000)
    parser.add_argument("--features-cache", type=str, default="data/features_cache.pt")
    parser.add_argument("--use-feature-cache", action="store_true", default=True)
    parser.add_argument("--no-feature-cache", action="store_false", dest="use_feature_cache")
    parser.add_argument(
        "--feature-extractor-kind",
        type=str,
        default="legacy_vggpp",
        choices=["legacy_vggpp", "vgg_plus_plus"],
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    return parser.parse_args()


def _atomic_save_results(out_path: Path, results: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)


def _load_or_init_results(out_path: Path, *, resume: bool) -> dict[str, Any]:
    if resume and out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    return {"_completed_jobs": []}


def _completed_jobs(results: dict[str, Any]) -> set[str]:
    jobs = results.get("_completed_jobs", [])
    if not isinstance(jobs, list):
        return set()
    return {str(x) for x in jobs}


def _mark_completed(results: dict[str, Any], job_id: str) -> None:
    jobs = results.setdefault("_completed_jobs", [])
    if job_id not in jobs:
        jobs.append(job_id)


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    out_path = Path(args.out)

    def log(msg: str) -> None:
        elapsed = time.perf_counter() - start_time
        print(f"[{elapsed:8.1f}s] {msg}", flush=True)

    def last_metrics(outputs: dict[str, object]) -> dict[str, float]:
        history = outputs.get("history", [])
        return history[-1] if history else {}

    def run_summary(outputs: dict[str, object]) -> dict[str, object]:
        last_step = int(outputs["step_metrics"][-1]["step"]) if outputs.get("step_metrics") else 0
        return {
            "steps_completed": last_step,
            "tasks_completed": len(outputs.get("anchor_task_accuracy", {})),
        }

    def recovery_metric(outputs: dict[str, object], *, first_n_examples: int = 100) -> float:
        return float(
            mean_recovery_window_accuracy(
                outputs.get("anchor_recovery", {}),
                first_n_examples=first_n_examples,
            )
        )

    results: dict[str, Any] = _load_or_init_results(out_path, resume=args.resume)
    completed_jobs = _completed_jobs(results)

    def persist(job_id: str) -> None:
        _mark_completed(results, job_id)
        _atomic_save_results(out_path, results)
        completed_jobs.add(job_id)

    def skip_or_start(job_id: str, label: str) -> bool:
        if job_id in completed_jobs:
            log(f"{label} (skip: already completed)")
            return True
        log(f"{label} (start)")
        return False

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
        log("Loading CIFAR-100 dataset...")
        dataset = load_cifar100(root=args.data_root, config=stream_cfg, train=True, download=True)
        labels = np.asarray(dataset.targets if hasattr(dataset, "targets") else dataset.labels, dtype=np.int64)
        images_np = dataset.data
        log(f"Dataset ready: {len(labels)} training samples.")

    base_training = dict(
        device=args.device,
        max_steps=args.max_steps,
        log_every=max(1, args.max_steps // 8),
        features_cache=args.features_cache,
        use_feature_cache=args.use_feature_cache,
    )

    claim4_experiments = [
        {
            "name": "full_method",
            "freeze_step": None,
            "context_len": 512,
            "n_streams": 20,
            "max_steps": args.max_steps,
        },
        {
            "name": "freeze_at_5k",
            "freeze_step": args.freeze_at_step,
            "context_len": 512,
            "n_streams": 20,
            "max_steps": args.max_steps,
        },
        {
            "name": "sgd_only",
            "freeze_step": None,
            "context_len": 0,
            "n_streams": 20,
            "max_steps": args.max_steps,
        },
    ]

    replay_values = [1, 2, 4, 8, 20]
    context_values = [1, 16, 64, 128, 256, 512]
    seeds = list(TrainingConfig().seeds)
    total_jobs = len(replay_values) + len(context_values) + len(claim4_experiments) + (2 * len(seeds)) + 1
    job_id = 0
    meta = results.setdefault("meta", {})
    meta.update(
        {
            "max_steps": args.max_steps,
            "features_cache": args.features_cache,
            "use_feature_cache": args.use_feature_cache,
            "resume_enabled": args.resume,
        }
    )
    _atomic_save_results(out_path, results)

    for replay_streams in replay_values:
        job_id += 1
        current_job_id = f"replay_E_{replay_streams}"
        if skip_or_start(current_job_id, f"[{job_id}/{total_jobs}] Replay ablation E={replay_streams}"):
            continue
        outputs = run_online_training(
            images_np=images_np,
            labels_np=labels,
            feature_bank=feature_bank,
            stream_config=stream_cfg,
            model_config=ModelConfig(context_window=512),
            training_config=TrainingConfig(replay_streams=replay_streams, **base_training),
            seed=args.seed,
            verbose=True,
            run_name=f"[replay E={replay_streams}]",
            feature_extractor_kind=args.feature_extractor_kind,
        )
        results[f"replay_E_{replay_streams}"] = last_metrics(outputs)
        results[f"replay_E_{replay_streams}"].update(run_summary(outputs))
        persist(current_job_id)
        log(f"[{job_id}/{total_jobs}] Replay ablation E={replay_streams} (done) {results[f'replay_E_{replay_streams}']}")

    for context_window in context_values:
        job_id += 1
        current_job_id = f"context_C_{context_window}"
        if skip_or_start(current_job_id, f"[{job_id}/{total_jobs}] Context ablation C={context_window}"):
            continue
        outputs = run_online_training(
            images_np=images_np,
            labels_np=labels,
            feature_bank=feature_bank,
            stream_config=stream_cfg,
            model_config=ModelConfig(context_window=context_window),
            training_config=TrainingConfig(**base_training),
            seed=args.seed,
            verbose=True,
            run_name=f"[context C={context_window}]",
            feature_extractor_kind=args.feature_extractor_kind,
        )
        results[f"context_C_{context_window}"] = last_metrics(outputs)
        results[f"context_C_{context_window}"].update(run_summary(outputs))
        results[f"context_C_{context_window}"]["recovery_window_accuracy_mean"] = recovery_metric(outputs)
        results[f"context_C_{context_window}"]["claim_metric_mean"] = results[f"context_C_{context_window}"]["recovery_window_accuracy_mean"]
        persist(current_job_id)
        log(f"[{job_id}/{total_jobs}] Context ablation C={context_window} (done) {results[f'context_C_{context_window}']}")

    claim4_results: dict[str, object] = {}
    for exp in claim4_experiments:
        job_id += 1
        current_job_id = f"claim4:{exp['name']}"
        if skip_or_start(current_job_id, f"[{job_id}/{total_jobs}] Claim4 {exp['name']}"):
            if isinstance(results.get("claim4"), dict) and exp["name"] in results["claim4"]:
                claim4_results[exp["name"]] = results["claim4"][exp["name"]]
            continue
        outputs = run_online_training(
            images_np=images_np,
            labels_np=labels,
            feature_bank=feature_bank,
            stream_config=stream_cfg,
            model_config=ModelConfig(context_window=int(exp["context_len"])),
            training_config=TrainingConfig(
                replay_streams=int(exp["n_streams"]),
                max_steps=int(exp["max_steps"]),
                device=args.device,
                log_every=max(1, int(exp["max_steps"]) // 8),
                features_cache=args.features_cache,
                use_feature_cache=args.use_feature_cache,
            ),
            seed=args.seed,
            model_kind="pi",
            freeze_after_step=exp["freeze_step"],
            verbose=True,
            run_name=f"[claim4 {exp['name']}]",
            feature_extractor_kind=args.feature_extractor_kind,
        )
        summary = run_summary(outputs)
        last = last_metrics(outputs)
        claim4_results[exp["name"]] = {
            "rolling_acc_mean": float(last.get("rolling_accuracy", 0.0)),
            "recovery_window_accuracy_mean": recovery_metric(outputs),
            "claim_metric_mean": recovery_metric(outputs),
            "instant_acc_last": float(last.get("instant_accuracy", 0.0)),
            "loss_last": float(last.get("loss", 0.0)),
            "steps_completed": int(summary["steps_completed"]),
            "tasks_completed": int(summary["tasks_completed"]),
        }
        results["claim4"] = claim4_results
        log(f"[{job_id}/{total_jobs}] Claim4 {exp['name']} (done) {claim4_results[exp['name']]}")
        if "freeze_at_5k" in claim4_results:
            results["freeze_weights"] = {
                "freeze_at_step": args.freeze_at_step,
                "last": {
                    "rolling_accuracy": claim4_results["freeze_at_5k"]["rolling_acc_mean"],
                    "instant_accuracy": claim4_results["freeze_at_5k"]["instant_acc_last"],
                    "loss": claim4_results["freeze_at_5k"]["loss_last"],
                },
            }
        persist(current_job_id)
    results["claim4"] = claim4_results
    if "freeze_at_5k" in claim4_results:
        results["freeze_weights"] = {
            "freeze_at_step": args.freeze_at_step,
            "last": {
                "rolling_accuracy": claim4_results["freeze_at_5k"]["rolling_acc_mean"],
                "instant_accuracy": claim4_results["freeze_at_5k"]["instant_acc_last"],
                "loss": claim4_results["freeze_at_5k"]["loss_last"],
            },
        }

    pi_vs_two_token: dict[str, object] = {}
    for model_kind in ("pi", "two_token"):
        per_seed: dict[str, object] = {}
        for seed in seeds:
            job_id += 1
            current_job_id = f"compare:{model_kind}:{seed}"
            if skip_or_start(current_job_id, f"[{job_id}/{total_jobs}] {model_kind} seed={seed}"):
                if (
                    isinstance(results.get("pi_vs_two_token"), dict)
                    and isinstance(results["pi_vs_two_token"].get(model_kind), dict)
                    and str(seed) in results["pi_vs_two_token"][model_kind]
                ):
                    per_seed[str(seed)] = results["pi_vs_two_token"][model_kind][str(seed)]
                continue
            outputs = run_online_training(
                images_np=images_np,
                labels_np=labels,
                feature_bank=feature_bank,
                stream_config=stream_cfg,
                model_config=ModelConfig(context_window=512),
                training_config=TrainingConfig(**base_training),
                seed=seed,
                model_kind=model_kind,
                verbose=True,
                run_name=f"[{model_kind} seed={seed}]",
                feature_extractor_kind=args.feature_extractor_kind,
            )
            per_seed[str(seed)] = last_metrics(outputs)
            per_seed[str(seed)].update(run_summary(outputs))
            results["pi_vs_two_token"] = {
                **results.get("pi_vs_two_token", {}),
                model_kind: per_seed,
            }
            persist(current_job_id)
            log(f"[{job_id}/{total_jobs}] {model_kind} seed={seed} (done) {per_seed[str(seed)]}")
        pi_vs_two_token[model_kind] = per_seed
    results["pi_vs_two_token"] = pi_vs_two_token

    job_id += 1
    current_job_id = "early_vs_late"
    if skip_or_start(current_job_id, f"[{job_id}/{total_jobs}] Early-vs-late run"):
        _atomic_save_results(out_path, results)
        log(f"All ablations done. Wrote {out_path}")
        return
    early_late = run_online_training(
        images_np=images_np,
        labels_np=labels,
        feature_bank=feature_bank,
        stream_config=stream_cfg,
        model_config=ModelConfig(context_window=512),
        training_config=TrainingConfig(**base_training),
        seed=args.seed,
        model_kind="pi",
        verbose=True,
        run_name="[early vs late]",
        feature_extractor_kind=args.feature_extractor_kind,
    )
    recovery = early_late.get("anchor_recovery", {})
    early_tasks = [recovery[str(i)] for i in range(0, 10) if str(i) in recovery]
    late_tasks = [recovery[str(i)] for i in range(80, 100) if str(i) in recovery]
    if early_tasks and late_tasks:
        early_acc = float(np.mean(np.array(early_tasks, dtype=np.float32)))
        late_acc = float(np.mean(np.array(late_tasks, dtype=np.float32)))
    else:
        hist = early_late.get("history", [])
        if hist:
            midpoint = max(len(hist) // 2, 1)
            early = hist[:midpoint]
            late = hist[midpoint:]
            early_acc = (sum(item["instant_accuracy"] for item in early) / len(early)) if early else 0.0
            late_acc = (sum(item["instant_accuracy"] for item in late) / len(late)) if late else early_acc
        else:
            early_acc = 0.0
            late_acc = 0.0
    results["early_vs_late"] = {
        "early_instant_accuracy_mean": early_acc,
        "late_instant_accuracy_mean": late_acc,
    }
    persist(current_job_id)
    log(f"[{job_id}/{total_jobs}] Early-vs-late run (done) {results['early_vs_late']}")

    _atomic_save_results(out_path, results)
    log(f"All ablations done. Wrote {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Previously completed jobs were already saved; rerun with the same --out path to resume.", flush=True)
        raise SystemExit(130)
