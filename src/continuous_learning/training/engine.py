from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from continuous_learning.config import ModelConfig, StreamConfig, TrainingConfig
from continuous_learning.data.cifar_stream import CIFARTaskStream
from continuous_learning.models import (
    OnlineSGDHead,
    PiTransformer,
    TwoTokenTransformerBaseline,
    VGGPlusPlusFeatureExtractor,
    VGGPlusPlusJoint,
)
from continuous_learning.models.pi_transformer import KVCache
from continuous_learning.training.metrics import OnlineMetricTracker
from continuous_learning.training.replay import ReplayStreamReader


@dataclass
class ModelStack:
    feature_extractor: nn.Module
    temporal_model: nn.Module
    optimizer: torch.optim.Optimizer


def build_default_model_stack(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    model_kind: str = "pi",
    use_feature_cache: bool = False,
    feature_extractor_kind: str = "legacy_vggpp",
) -> ModelStack:
    feature_extractor: nn.Module
    if use_feature_cache:
        feature_extractor = nn.Identity()
    elif feature_extractor_kind == "vgg_plus_plus":
        feature_extractor = VGGPlusPlusJoint(output_dim=model_config.feature_dim)
    else:
        feature_extractor = VGGPlusPlusFeatureExtractor(output_dim=model_config.feature_dim)
    if model_kind == "pi":
        temporal_model: nn.Module = PiTransformer(model_config)
    elif model_kind == "two_token":
        temporal_model = TwoTokenTransformerBaseline(model_config)
    elif model_kind == "online_sgd":
        temporal_model = OnlineSGDHead(feature_dim=model_config.feature_dim, num_classes=model_config.num_classes)
    else:
        raise ValueError(f"Unknown model_kind={model_kind}")

    params = list(feature_extractor.parameters()) + list(temporal_model.parameters())
    optimizer = AdamW(
        params=params,
        lr=training_config.learning_rate,
        betas=(training_config.beta1, training_config.beta2),
        eps=training_config.eps,
        weight_decay=training_config.weight_decay,
    )
    return ModelStack(
        feature_extractor=feature_extractor,
        temporal_model=temporal_model,
        optimizer=optimizer,
    )


def _make_replay_readers(
    task_indices: list[np.ndarray],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    seed: int,
    anchor_stream_no_reset: bool,
) -> list[ReplayStreamReader]:
    readers: list[ReplayStreamReader] = []
    for i in range(training_config.replay_streams):
        reader = ReplayStreamReader(
            task_stream=task_indices,
            model_config=model_config,
            seed=seed + i * 17,
            current_task=i % max(len(task_indices), 1),
            allow_reset=not (anchor_stream_no_reset and i == 0),
        )
        readers.append(reader)
    return readers


def _normalize_images(
    images: torch.Tensor,
    stream_config: StreamConfig,
    device: torch.device,
) -> torch.Tensor:
    images = images.to(device).float() / 255.0
    images = images.permute(0, 3, 1, 2)
    mean = torch.tensor(stream_config.cifar_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(stream_config.cifar_std, device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def _load_feature_bank(training_config: TrainingConfig) -> torch.Tensor | None:
    if not training_config.use_feature_cache:
        return None
    path = training_config.features_cache
    if not path:
        return None
    try:
        cache = torch.load(path, map_location="cpu")
    except FileNotFoundError:
        return None
    feats = cache.get("features")
    if feats is None:
        return None
    return feats.contiguous()


@torch.no_grad()
def _evaluate_final_task_accuracies(
    *,
    feature_extractor: nn.Module,
    temporal_model: nn.Module,
    model_kind: str,
    images_np: np.ndarray | None,
    feature_bank: torch.Tensor | None,
    labels_np: np.ndarray,
    task_indices: list[np.ndarray],
    stream_config: StreamConfig,
    model_config: ModelConfig,
    device: torch.device,
    final_eval_max_examples_per_task: int | None,
) -> dict[int, float]:
    feature_extractor.eval()
    temporal_model.eval()

    out: dict[int, float] = {}
    for task_id, indices in enumerate(task_indices):
        if final_eval_max_examples_per_task is not None and final_eval_max_examples_per_task > 0:
            indices = indices[:final_eval_max_examples_per_task]
        total = 0
        correct = 0
        caches: list[KVCache | None] = [None] * model_config.n_blocks

        for start in range(0, len(indices), model_config.chunk_size):
            stop = min(start + model_config.chunk_size, len(indices))
            chunk_idx = indices[start:stop]
            batch_labels = torch.from_numpy(labels_np[chunk_idx]).to(device=device, dtype=torch.long)
            if feature_bank is not None:
                features = feature_bank[torch.as_tensor(chunk_idx, dtype=torch.long)].to(device=device, dtype=torch.float32).unsqueeze(0)
            else:
                if images_np is None:
                    raise ValueError("images_np must be provided when feature cache is disabled.")
                batch_images = torch.from_numpy(images_np[chunk_idx])
                x = _normalize_images(batch_images, stream_config=stream_config, device=device)
                features = feature_extractor(x).unsqueeze(0)
            labels_seq = batch_labels.unsqueeze(0)
            logits, next_cache = temporal_model(features=features, labels=labels_seq, caches=caches)
            if model_kind in {"pi", "two_token"}:
                caches = next_cache

            preds = logits.view(-1, model_config.num_classes).argmax(dim=-1)
            correct += int((preds == batch_labels).sum().item())
            total += int(batch_labels.numel())

        out[task_id] = float(correct / max(total, 1))

    feature_extractor.train()
    temporal_model.train()
    return out


def run_online_training(
    images_np: np.ndarray | None,
    labels_np: np.ndarray,
    stream_config: StreamConfig = StreamConfig(),
    model_config: ModelConfig = ModelConfig(),
    training_config: TrainingConfig = TrainingConfig(),
    seed: int = 42,
    model_kind: str = "pi",
    freeze_after_step: int | None = None,
    verbose: bool = False,
    run_name: str | None = None,
    collect_diagnostics: bool = True,
    anchor_stream_no_reset: bool = True,
    recovery_examples: int = 200,
    compute_final_task_eval: bool = False,
    final_eval_max_examples_per_task: int | None = None,
    feature_bank: torch.Tensor | None = None,
    feature_extractor_kind: str = "legacy_vggpp",
    checkpoint_path: str | None = None,
    checkpoint_every: int = 0,
    resume: bool = True,
) -> dict[str, object]:
    """
    Executes the core online loop with replay streams:
    - E replay streams share model weights
    - each stream has its own sequence state + KV cache
    - reset probability follows bernoulli(S / t)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(training_config.device)
    local_feature_bank = feature_bank if feature_bank is not None else _load_feature_bank(training_config)
    use_feature_cache = local_feature_bank is not None
    if not use_feature_cache and training_config.use_feature_cache and verbose:
        name = f"{run_name} " if run_name else ""
        print(f"{name}feature cache requested but not found at '{training_config.features_cache}', using live VGG++.", flush=True)
    if not use_feature_cache and images_np is None:
        raise ValueError("images_np is required when feature cache is unavailable.")
    task_stream = CIFARTaskStream(labels=labels_np, config=stream_config, num_total_classes=100)
    task_indices = task_stream.build_task_index_stream()

    stack = build_default_model_stack(
        model_config=model_config,
        training_config=training_config,
        model_kind=model_kind,
        use_feature_cache=use_feature_cache,
        feature_extractor_kind=feature_extractor_kind,
    )
    feature_extractor = stack.feature_extractor.to(device)
    temporal_model = stack.temporal_model.to(device)
    optimizer = stack.optimizer

    metric_tracker = OnlineMetricTracker(rolling_window=training_config.rolling_window)
    replay_readers = _make_replay_readers(
        task_indices=task_indices,
        model_config=model_config,
        training_config=training_config,
        seed=seed,
        anchor_stream_no_reset=anchor_stream_no_reset,
    )
    if model_kind in {"pi", "two_token"}:
        caches: list[list[KVCache | None] | list[object]] = [
            [None] * model_config.n_blocks for _ in range(training_config.replay_streams)
        ]
    else:
        caches = [[] for _ in range(training_config.replay_streams)]

    feature_extractor.train()
    temporal_model.train()
    history: list[dict[str, float]] = []
    frozen = False
    step_metrics: list[dict[str, float | int]] = []
    anchor_recent = deque()
    rolling_steps = max(1, training_config.rolling_window // model_config.chunk_size)
    anchor_task_correct: dict[int, int] = {}
    anchor_task_total: dict[int, int] = {}
    recovery_sum: dict[int, list[float]] = {}
    recovery_count: dict[int, list[int]] = {}
    prev_anchor_task: int | None = None
    task_switch_examples: list[int] = []
    start_step = 1
    best_anchor_roll = float("-inf")

    def _checkpoint_payload(step: int) -> dict[str, object]:
        cache_payload: list[list[dict[str, torch.Tensor] | None]] = []
        if model_kind in {"pi", "two_token"}:
            for stream_cache in caches:
                stream_payload: list[dict[str, torch.Tensor] | None] = []
                for layer_cache in stream_cache:
                    if layer_cache is None:
                        stream_payload.append(None)
                    else:
                        stream_payload.append(
                            {
                                "key": layer_cache.key.detach().cpu(),
                                "value": layer_cache.value.detach().cpu(),
                            }
                        )
                cache_payload.append(stream_payload)
        replay_states = [
            {
                "current_task": int(reader.current_task),
                "position": int(reader.position),
            }
            for reader in replay_readers
        ]
        return {
            "step": int(step),
            "feature_extractor": feature_extractor.state_dict(),
            "temporal_model": temporal_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "replay_states": replay_states,
            "caches": cache_payload,
            "history": history,
            "step_metrics": step_metrics,
            "anchor_task_correct": anchor_task_correct,
            "anchor_task_total": anchor_task_total,
            "recovery_sum": recovery_sum,
            "recovery_count": recovery_count,
            "task_switch_examples": task_switch_examples,
            "prev_anchor_task": prev_anchor_task,
            "frozen": frozen,
            "anchor_recent": list(anchor_recent),
            "best_anchor_roll": best_anchor_roll,
        }

    if checkpoint_path and resume and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        feature_extractor.load_state_dict(ckpt["feature_extractor"])
        temporal_model.load_state_dict(ckpt["temporal_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0)) + 1

        replay_states = ckpt.get("replay_states", [])
        if isinstance(replay_states, list):
            for reader, state in zip(replay_readers, replay_states, strict=False):
                if isinstance(state, dict):
                    reader.current_task = int(state.get("current_task", reader.current_task))
                    reader.position = int(state.get("position", reader.position))

        if model_kind in {"pi", "two_token"}:
            cache_payload = ckpt.get("caches", [])
            if isinstance(cache_payload, list):
                restored: list[list[KVCache | None]] = []
                for stream in cache_payload:
                    stream_out: list[KVCache | None] = []
                    if isinstance(stream, list):
                        for layer in stream:
                            if layer is None:
                                stream_out.append(None)
                            else:
                                stream_out.append(
                                    KVCache(
                                        key=layer["key"].to(device),
                                        value=layer["value"].to(device),
                                    )
                                )
                    restored.append(stream_out)
                if len(restored) == len(caches):
                    caches = restored

        history = list(ckpt.get("history", history))
        step_metrics = list(ckpt.get("step_metrics", step_metrics))
        anchor_task_correct = dict(ckpt.get("anchor_task_correct", anchor_task_correct))
        anchor_task_total = dict(ckpt.get("anchor_task_total", anchor_task_total))
        recovery_sum = dict(ckpt.get("recovery_sum", recovery_sum))
        recovery_count = dict(ckpt.get("recovery_count", recovery_count))
        task_switch_examples = list(ckpt.get("task_switch_examples", task_switch_examples))
        prev_anchor_task = ckpt.get("prev_anchor_task", prev_anchor_task)
        frozen = bool(ckpt.get("frozen", frozen))
        best_anchor_roll = float(ckpt.get("best_anchor_roll", best_anchor_roll))
        anchor_recent = deque(ckpt.get("anchor_recent", []), maxlen=rolling_steps)
        if verbose:
            name = f"{run_name} " if run_name else ""
            print(f"{name}Resumed from {checkpoint_path} at step {start_step}.", flush=True)

    for step in range(start_step, training_config.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)
        anchor_task_id = -1
        anchor_instant_acc = 0.0
        anchor_position_start = 0

        for stream_id, reader in enumerate(replay_readers):
            chunk_indices = reader.next_chunk(step)
            chunk_info = reader.last_chunk_info or {}
            batch_labels = torch.from_numpy(labels_np[chunk_indices]).to(device=device, dtype=torch.long)
            if use_feature_cache:
                idx = torch.as_tensor(chunk_indices, dtype=torch.long)
                features = local_feature_bank[idx].to(device=device, dtype=torch.float32).unsqueeze(0)
            else:
                if images_np is None:
                    raise ValueError("images_np is required when feature cache is unavailable.")
                batch_images = torch.from_numpy(images_np[chunk_indices])
                x = _normalize_images(batch_images, stream_config=stream_config, device=device)
                features = feature_extractor(x).unsqueeze(0)  # [1, S, D]
            labels_seq = batch_labels.unsqueeze(0)  # [1, S]

            logits, next_cache = temporal_model(
                features=features,
                labels=labels_seq,
                caches=caches[stream_id],
            )
            # Cache is detached inside attention; keep per-stream state.
            if model_kind in {"pi", "two_token"}:
                caches[stream_id] = next_cache

            loss = F.cross_entropy(
                logits.view(-1, model_config.num_classes),
                batch_labels,
            )
            total_loss = total_loss + loss
            metrics = metric_tracker.update(reader.current_task, logits.view(-1, model_config.num_classes), batch_labels)

            if stream_id == 0:
                preds = logits.view(-1, model_config.num_classes).argmax(dim=-1)
                correct_mask = (preds == batch_labels)
                anchor_instant_acc = float(correct_mask.float().mean().item())
                anchor_task_id = int(chunk_info.get("task_id", reader.current_task))
                anchor_position_start = int(chunk_info.get("position_start", 0))

                anchor_task_correct[anchor_task_id] = anchor_task_correct.get(anchor_task_id, 0) + int(correct_mask.sum().item())
                anchor_task_total[anchor_task_id] = anchor_task_total.get(anchor_task_id, 0) + int(correct_mask.numel())

                if anchor_task_id not in recovery_sum:
                    recovery_sum[anchor_task_id] = [0.0] * recovery_examples
                    recovery_count[anchor_task_id] = [0] * recovery_examples
                for offset, ok in enumerate(correct_mask.detach().cpu().tolist()):
                    pos = anchor_position_start + offset
                    if pos < recovery_examples:
                        recovery_sum[anchor_task_id][pos] += float(ok)
                        recovery_count[anchor_task_id][pos] += 1

        total_loss = total_loss / training_config.replay_streams
        if freeze_after_step is not None and step >= freeze_after_step and not frozen:
            for p in feature_extractor.parameters():
                p.requires_grad_(False)
            for p in temporal_model.parameters():
                p.requires_grad_(False)
            frozen = True

        if not frozen:
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(feature_extractor.parameters()) + list(temporal_model.parameters()),
                training_config.grad_clip_max_norm,
            )
            optimizer.step()

        if collect_diagnostics:
            anchor_recent.append(anchor_instant_acc)
            if len(anchor_recent) > rolling_steps:
                anchor_recent.popleft()
            anchor_roll = float(sum(anchor_recent) / len(anchor_recent)) if anchor_recent else 0.0

            global_examples = step * model_config.chunk_size
            if prev_anchor_task is None or anchor_task_id != prev_anchor_task:
                task_switch_examples.append(global_examples)
            prev_anchor_task = anchor_task_id

            step_metrics.append(
                {
                    "step": step,
                    "global_examples": global_examples,
                    "task_id": anchor_task_id,
                    "position_start": anchor_position_start,
                    "loss": float(total_loss.item()),
                    "instant_accuracy": anchor_instant_acc,
                    "rolling_accuracy": anchor_roll,
                }
            )

        if checkpoint_path and checkpoint_every > 0 and step % checkpoint_every == 0:
            ckpt_path = Path(checkpoint_path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(_checkpoint_payload(step), ckpt_path)
            current_roll = 0.0
            if collect_diagnostics and step_metrics:
                current_roll = float(step_metrics[-1]["rolling_accuracy"])
            elif metric_tracker.rolling_buffer:
                current_roll = float(sum(metric_tracker.rolling_buffer) / len(metric_tracker.rolling_buffer))
            if current_roll > best_anchor_roll:
                best_anchor_roll = current_roll
                best_path = ckpt_path.parent / "best.pt"
                torch.save(_checkpoint_payload(step), best_path)

        if step % training_config.log_every == 0:
            checkpoint = {
                "step": float(step),
                "loss": float(total_loss.item()),
                "instant_accuracy": metrics["instant_accuracy"],
                "rolling_accuracy": metrics["rolling_accuracy"],
            }
            history.append(checkpoint)
            if verbose:
                name = f"{run_name} " if run_name else ""
                print(
                    f"{name}step={step}/{training_config.max_steps} "
                    f"loss={checkpoint['loss']:.4f} "
                    f"inst_acc={checkpoint['instant_accuracy']:.4f} "
                    f"roll_acc={checkpoint['rolling_accuracy']:.4f}",
                    flush=True,
                )

    if checkpoint_path and checkpoint_every > 0:
        ckpt_path = Path(checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_checkpoint_payload(training_config.max_steps), ckpt_path)

    return {
        "history": history,
        "accuracy_vs_task": metric_tracker.accuracy_vs_task_number(),
        "recovery_speed": metric_tracker.recovery_speed(),
        "frozen_at_step": float(freeze_after_step) if freeze_after_step is not None else None,
        "model_kind": model_kind,
        "step_metrics": step_metrics,
        "anchor_task_accuracy": {
            int(task_id): float(anchor_task_correct[task_id] / max(anchor_task_total.get(task_id, 1), 1))
            for task_id in sorted(anchor_task_correct.keys())
        },
        "anchor_recovery": {
            int(task_id): [
                float(recovery_sum[task_id][i] / recovery_count[task_id][i]) if recovery_count[task_id][i] else 0.0
                for i in range(recovery_examples)
            ]
            for task_id in sorted(recovery_sum.keys())
        },
        "task_switch_examples": task_switch_examples,
        "task_size_examples": stream_config.examples_per_task,
        "chunk_size": model_config.chunk_size,
        "final_task_accuracy": _evaluate_final_task_accuracies(
            feature_extractor=feature_extractor,
            temporal_model=temporal_model,
            model_kind=model_kind,
            images_np=images_np,
            feature_bank=local_feature_bank,
            labels_np=labels_np,
            task_indices=task_indices,
            stream_config=stream_config,
            model_config=model_config,
            device=device,
            final_eval_max_examples_per_task=final_eval_max_examples_per_task,
        )
        if compute_final_task_eval
        else {},
    }
