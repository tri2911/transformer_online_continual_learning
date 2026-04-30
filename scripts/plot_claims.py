from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot claim-mapped charts from ablation JSON")
    parser.add_argument("--results", type=str, required=True, help="Path to ablation_results.json")
    parser.add_argument("--out-dir", type=str, default="outputs/pretrained/figures")
    return parser.parse_args()


def _save_bar(data: dict[str, float], title: str, out_path: Path, ylabel: str) -> None:
    keys = list(data.keys())
    vals = [data[k] for k in keys]
    plt.figure(figsize=(8, 4))
    plt.bar(keys, vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _metric_value(entry: dict[str, object], *, fallback_key: str) -> float:
    if "claim_metric_mean" in entry:
        return float(entry["claim_metric_mean"])
    if "recovery_window_accuracy_mean" in entry:
        return float(entry["recovery_window_accuracy_mean"])
    return float(entry.get(fallback_key, 0.0))


def _plot_claim4_three_conditions(results: dict[str, object], out_path: Path) -> None:
    claim4 = results.get("claim4", {})
    conditions = {
        "full_method": "Full Method\n(ICL + SGD)",
        "freeze_at_5k": "Freeze Weights\nat Step 5K\n(ICL only after)",
        "sgd_only": "No Context\n(SGD only, C=0)",
    }
    keys = list(conditions.keys())
    vals = [_metric_value(claim4.get(k, {}), fallback_key="rolling_acc_mean") for k in keys]
    names = [conditions[k] for k in keys]
    colors = ["#2196F3", "#FF9800", "#9E9E9E"]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, vals, color=colors, width=0.55)
    for bar, val in zip(bars, vals, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.title("CLAIM 4: Both ICL and In-Weight Learning Are Necessary")
    plt.ylabel("Mean Rolling Accuracy")
    plt.ylim(0, 1.0)
    plt.annotate(
        "Expected: Full Method > Freeze > SGD only",
        xy=(0.5, 0.02),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
        color="grey",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))

    replay = {
        k: float(v.get("rolling_accuracy", 0.0))
        for k, v in results.items()
        if k.startswith("replay_E_")
    }
    if replay:
        _save_bar(
            replay,
            "CLAIM 2: Replay Streams Ablation",
            out_dir / "claim2_replay_ablation.png",
            "Rolling Accuracy",
        )

    context = {
        k: _metric_value(v, fallback_key="rolling_accuracy")
        for k, v in results.items()
        if k.startswith("context_C_")
    }
    if context:
        _save_bar(
            context,
            "CLAIM 3: Context Window Ablation",
            out_dir / "claim3_context_ablation.png",
            "Mean Accuracy In First 100 Examples Per Task",
        )

    if isinstance(results.get("claim4"), dict):
        _plot_claim4_three_conditions(results, out_dir / "claim4_freeze_weights.png")
    else:
        freeze_last = float(results.get("freeze_weights", {}).get("last", {}).get("rolling_accuracy", 0.0))
        _save_bar(
            {"freeze_run": freeze_last},
            "CLAIM 4: Freeze-Weights Synergy Check",
            out_dir / "claim4_freeze_weights.png",
            "Rolling Accuracy",
        )

    compare = {}
    for model_kind, per_seed in results.get("pi_vs_two_token", {}).items():
        vals = [float(v.get("rolling_accuracy", 0.0)) for v in per_seed.values()]
        compare[model_kind] = sum(vals) / max(len(vals), 1)
    if compare:
        _save_bar(
            compare,
            "CLAIM 5: Pi-Transformer vs 2-Token",
            out_dir / "claim5_pi_vs_two_token.png",
            "Mean Rolling Accuracy",
        )

    early_late = results.get("early_vs_late", {})
    _save_bar(
        {
            "early": float(early_late.get("early_instant_accuracy_mean", 0.0)),
            "late": float(early_late.get("late_instant_accuracy_mean", 0.0)),
        },
        "CLAIM 6: Early vs Late Adaptation",
        out_dir / "claim6_early_vs_late.png",
        "Instant Accuracy",
    )

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
