from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pi-Transformer on CIFAR-100 stream")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--replay-streams", type=int, default=20)
    parser.add_argument("--context-window", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-kind", type=str, default="pi", choices=["pi", "two_token"])
    parser.add_argument("--freeze-after-step", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--features-cache", type=str, default="data/features_cache.pt")
    parser.add_argument("--use-feature-cache", action="store_true", default=True)
    parser.add_argument("--no-feature-cache", action="store_false", dest="use_feature_cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    def log(msg: str) -> None:
        elapsed = time.perf_counter() - start_time
        print(f"[{elapsed:8.1f}s] {msg}", flush=True)

    stream_cfg = StreamConfig()
    model_cfg = ModelConfig(context_window=args.context_window)
    training_cfg = TrainingConfig(
        replay_streams=args.replay_streams,
        max_steps=args.max_steps,
        device=args.device,
        log_every=args.log_every,
        features_cache=args.features_cache,
        use_feature_cache=args.use_feature_cache,
    )

    images_np: np.ndarray | None = None
    feature_bank: torch.Tensor | None = None
    labels: np.ndarray
    if args.use_feature_cache and pathlib.Path(args.features_cache).exists():
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
    log(
        "Training start: "
        f"model={args.model_kind}, device={args.device}, "
        f"steps={args.max_steps}, replay_streams={args.replay_streams}, "
        f"context_window={args.context_window}"
    )
    outputs = run_online_training(
        images_np=images_np,
        labels_np=labels,
        feature_bank=feature_bank,
        stream_config=stream_cfg,
        model_config=model_cfg,
        training_config=training_cfg,
        seed=args.seed,
        model_kind=args.model_kind,
        freeze_after_step=args.freeze_after_step,
        verbose=True,
        run_name=f"[train/{args.model_kind}]",
    )
    log("Training finished.")
    log(f"Logged checkpoints: {len(outputs['history'])}")
    if outputs["history"]:
        log(f"Last checkpoint: {outputs['history'][-1]}")


if __name__ == "__main__":
    main()
