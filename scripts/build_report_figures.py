from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build report-ready figures and tables from report runs.")
    parser.add_argument("--runs-dir", type=str, default="outputs/pretrained/runs")
    parser.add_argument("--out-dir", type=str, default="outputs/pretrained/report")
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="",
        help="Optional directory for figure PNGs. If empty, uses --out-dir.",
    )
    parser.add_argument("--ablation-results", type=str, default="")
    return parser.parse_args()


METHOD_ORDER = [
    "online_sgd",
    "experience_replay",
    "pi_transformer",
    "two_token_transformer",
]

METHOD_STYLE = {
    "online_sgd": {"color": "#6f6f6f", "linestyle": "--"},
    "experience_replay": {"color": "#1f77b4", "linestyle": "-"},
    "pi_transformer": {"color": "#2ca02c", "linestyle": "-"},
    "two_token_transformer": {"color": "#d62728", "linestyle": "-"},
}


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _load_runs(raw_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def _to_task_map(d: dict[str, Any] | dict[int, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for k, v in d.items():
        out[int(k)] = float(v)
    return out


def _to_task_curve_map(d: dict[str, Any] | dict[int, Any]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for k, v in d.items():
        out[int(k)] = [float(x) for x in v]
    return out


def _series_from_run(run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = run["result"].get("step_metrics", [])
    x = np.array([int(row["global_examples"]) for row in steps], dtype=np.int64)
    inst = np.array([float(row["instant_accuracy"]) for row in steps], dtype=np.float32)
    roll = np.array([float(row["rolling_accuracy"]) for row in steps], dtype=np.float32)
    return x, inst, roll


def _group_by_method(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        out.setdefault(run["method"], []).append(run)
    return out


def _aggregate_curves(method_runs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    series = [_series_from_run(r) for r in method_runs]
    min_len = min(len(x) for x, _, _ in series)
    x = series[0][0][:min_len]
    inst = np.stack([s[1][:min_len] for s in series], axis=0)
    roll = np.stack([s[2][:min_len] for s in series], axis=0)
    return x, inst.mean(0), inst.std(0), roll.mean(0), roll.std(0)


def _write_comparison_tables(grouped: dict[str, list[dict[str, Any]]], out_dir: Path) -> dict[str, dict[str, float]]:
    method_summary: dict[str, dict[str, float]] = {}
    for method, runs in grouped.items():
        rows = []
        for run in runs:
            result = run["result"]
            steps = result.get("step_metrics", [])
            task_map = _to_task_map(result.get("anchor_task_accuracy", {}))
            final_map = _to_task_map(result.get("final_task_accuracy", {}))
            avg_all = _safe_mean([float(x["instant_accuracy"]) for x in steps])
            last20 = [task_map[t] for t in range(80, 100) if t in task_map]
            avg_last20 = _safe_mean(last20)
            bwt_terms = [
                final_map[t] - task_map[t]
                for t in range(0, 99)
                if t in final_map and t in task_map
            ]
            bwt = _safe_mean(bwt_terms)
            recovery = _to_task_curve_map(result.get("anchor_recovery", {}))
            first100_vals: list[float] = []
            for curve in recovery.values():
                if not curve:
                    continue
                first100_vals.extend(curve[:100])
            # Random baseline for 10-way classification.
            fwt = _safe_mean(first100_vals) - 0.10
            rows.append((avg_all, avg_last20, bwt, fwt))
        method_summary[method] = {
            "avg_acc_all_tasks": _safe_mean([r[0] for r in rows]),
            "avg_acc_last_20_tasks": _safe_mean([r[1] for r in rows]),
            "bwt": _safe_mean([r[2] for r in rows]),
            "fwt": _safe_mean([r[3] for r in rows]),
        }

    csv_path = out_dir / "comparison_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Avg Acc (all tasks)", "Avg Acc (last 20 tasks)", "BWT (forgetting)", "FWT"])
        for method in sorted(method_summary.keys()):
            s = method_summary[method]
            writer.writerow(
                [
                    method,
                    f"{s['avg_acc_all_tasks']:.4f}",
                    f"{s['avg_acc_last_20_tasks']:.4f}",
                    f"{s['bwt']:.4f}",
                    f"{s['fwt']:.4f}",
                ]
            )

    md_path = out_dir / "comparison_table.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Method | Avg Acc (all tasks) | Avg Acc (last 20 tasks) | BWT (forgetting) | FWT |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for method in sorted(method_summary.keys()):
            s = method_summary[method]
            f.write(
                f"| {method} | {s['avg_acc_all_tasks']:.4f} | {s['avg_acc_last_20_tasks']:.4f} | {s['bwt']:.4f} | {s['fwt']:.4f} |\n"
            )
    return method_summary


def _extract_early_task_curve(
    run: dict[str, Any],
    *,
    early_task_max: int = 9,
    rolling_window: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    steps = run["result"].get("step_metrics", [])
    xs: list[int] = []
    ys: list[float] = []
    for row in steps:
        task_id = int(row.get("task_id", -1))
        if 0 <= task_id <= early_task_max:
            xs.append(int(row.get("global_examples", 0)))
            ys.append(float(row.get("instant_accuracy", 0.0)))

    if not xs:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    values = np.asarray(ys, dtype=np.float32)
    if rolling_window <= 1:
        return np.asarray(xs, dtype=np.int64), values

    window = min(rolling_window, len(values))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    rolled = np.convolve(values, kernel, mode="same").astype(np.float32)
    return np.asarray(xs, dtype=np.int64), rolled


def _extract_early_task_curve_full_stream(
    run: dict[str, Any],
    *,
    early_task_max: int = 9,
    rolling_window: int = 200,
    transition_points: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a full-stream early-task retention curve from existing diagnostics.

    Available run logs contain online accuracy for the current anchor task only.
    We therefore use:
    - online early-task accuracy while anchor task is in [0..early_task_max]
    - final evaluated early-task accuracy as the post-early-training retention level
    """
    steps = run["result"].get("step_metrics", [])
    if not steps:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    x_early, y_early = _extract_early_task_curve(
        run,
        early_task_max=early_task_max,
        rolling_window=rolling_window,
    )
    if x_early.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    max_x = int(max(int(row.get("global_examples", 0)) for row in steps))
    final_task_accuracy = run["result"].get("final_task_accuracy", {})
    final_early_vals: list[float] = []
    for task_id in range(early_task_max + 1):
        if str(task_id) in final_task_accuracy:
            final_early_vals.append(float(final_task_accuracy[str(task_id)]))
        elif task_id in final_task_accuracy:
            final_early_vals.append(float(final_task_accuracy[task_id]))
    if not final_early_vals:
        final_early = float(y_early[-1])
    else:
        final_early = float(sum(final_early_vals) / len(final_early_vals))

    # Smoothly transition from the last online-early estimate to final retention.
    x_start = int(x_early[-1])
    if max_x <= x_start:
        return x_early, y_early

    n = max(2, int(transition_points))
    x_tail = np.linspace(x_start, max_x, n, dtype=np.int64)
    y_tail = np.linspace(float(y_early[-1]), final_early, n, dtype=np.float32)

    x_full = np.concatenate([x_early, x_tail[1:]], axis=0)
    y_full = np.concatenate([y_early, y_tail[1:]], axis=0)
    return x_full, y_full


def _catastrophic_bwt_payload(
    method_summary: dict[str, dict[str, float]],
) -> tuple[list[str], np.ndarray, list[str]]:
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for method in METHOD_ORDER:
        if method not in method_summary:
            continue
        bwt = float(method_summary[method]["bwt"])
        labels.append(method)
        values.append(bwt)
        colors.append("#d62728" if bwt < 0.0 else "#2ca02c")
    return labels, np.asarray(values, dtype=np.float32), colors


def _plot_main_accuracy(grouped: dict[str, list[dict[str, Any]]], figures_dir: Path) -> None:
    plt.figure(figsize=(12, 6))
    max_x = 0
    task_size = None
    for method in METHOD_ORDER:
        if method not in grouped:
            continue
        x, inst_m, _, roll_m, _ = _aggregate_curves(grouped[method])
        max_x = max(max_x, int(x[-1]) if len(x) else 0)
        if task_size is None and grouped[method]:
            task_size = int(grouped[method][0]["result"].get("task_size_examples", 5000))
        style = METHOD_STYLE[method]
        color = style["color"]
        linestyle = style["linestyle"]
        plt.plot(
            x,
            inst_m,
            color=color,
            linestyle=linestyle,
            alpha=0.25,
            linewidth=1.2,
            label=f"{method} instantaneous",
        )
        plt.plot(
            x,
            roll_m,
            color=color,
            linestyle=linestyle,
            alpha=0.95,
            linewidth=2.0,
            label=f"{method} rolling(500)",
        )

    if task_size and max_x:
        for xline in range(task_size, max_x + 1, task_size):
            plt.axvline(xline, color="#808080", linestyle="-", alpha=0.05, linewidth=0.8)

    plt.title("Main Accuracy Curve")
    plt.xlabel("Global timestep (examples seen)")
    plt.ylabel("Accuracy")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "figure1_main_accuracy_curve.png", dpi=180)
    plt.close()


def _plot_per_task_curve(grouped: dict[str, list[dict[str, Any]]], figures_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    for method in METHOD_ORDER:
        if method not in grouped:
            continue
        runs = grouped[method]
        mats = []
        for run in runs:
            task_map = _to_task_map(run["result"].get("anchor_task_accuracy", {}))
            vec = np.full(100, np.nan, dtype=np.float32)
            for t, v in task_map.items():
                if 0 <= t < 100:
                    vec[t] = v
            mats.append(vec)
        mat = np.stack(mats, axis=0)
        count = np.sum(~np.isnan(mat), axis=0)
        mean = np.divide(
            np.nansum(mat, axis=0),
            count,
            out=np.full(100, np.nan, dtype=np.float32),
            where=count > 0,
        )
        centered = np.where(np.isnan(mat), 0.0, mat - mean[None, :])
        var = np.divide(
            np.sum(centered * centered, axis=0),
            count,
            out=np.zeros(100, dtype=np.float32),
            where=count > 0,
        )
        std = np.sqrt(var)
        x = np.arange(100)
        style = METHOD_STYLE[method]
        plt.plot(
            x,
            mean,
            linewidth=2.0,
            color=style["color"],
            linestyle=style["linestyle"],
            label=method,
        )
        plt.fill_between(x, mean - std, mean + std, alpha=0.15, color=style["color"])

    plt.title("Per-Task Accuracy Curve (Forward Transfer)")
    plt.xlabel("Task number")
    plt.ylabel("Mean accuracy within task")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "figure2_per_task_accuracy_curve.png", dpi=180)
    plt.close()


def _plot_recovery_detail(grouped: dict[str, list[dict[str, Any]]], figures_dir: Path) -> None:
    if "pi_transformer" not in grouped:
        return
    run = grouped["pi_transformer"][0]
    recovery = _to_task_curve_map(run["result"].get("anchor_recovery", {}))
    if not recovery:
        return

    early_tasks = [t for t in range(0, 10) if t in recovery]
    late_tasks = [t for t in range(80, 100) if t in recovery]
    if not early_tasks or not late_tasks:
        return

    early_mat = np.stack([np.array(recovery[t], dtype=np.float32) for t in early_tasks], axis=0)
    late_mat = np.stack([np.array(recovery[t], dtype=np.float32) for t in late_tasks], axis=0)

    x = np.arange(early_mat.shape[1])
    plt.figure(figsize=(10, 5))
    plt.plot(x, early_mat.mean(0), label="Early tasks (1-10)", linewidth=2.2)
    plt.plot(x, late_mat.mean(0), label="Late tasks (80-100)", linewidth=2.2)
    plt.title("Within-Task Recovery Detail")
    plt.xlabel("Position within task (example index)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "figure5_within_task_recovery.png", dpi=180)
    plt.close()


def _plot_seed_stability(grouped: dict[str, list[dict[str, Any]]], figures_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    plotted = False
    for method in ("pi_transformer", "two_token_transformer"):
        if method not in grouped or len(grouped[method]) < 2:
            continue
        x, _, _, roll_mean, roll_std = _aggregate_curves(grouped[method])
        color = METHOD_STYLE[method]["color"]
        plt.plot(x, roll_mean, linewidth=2.2, color=color, label=f"{method} mean")
        plt.fill_between(x, roll_mean - roll_std, roll_mean + roll_std, alpha=0.2, color=color, label=f"{method} ±std")
        plotted = True
    if not plotted:
        plt.close()
        return

    plt.title("Training Stability (Seed Variance)")
    plt.xlabel("Global timestep (examples seen)")
    plt.ylabel("Rolling accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "figure6_claim5_stability.png", dpi=180)
    plt.close()


def _plot_catastrophic_forgetting(
    grouped: dict[str, list[dict[str, Any]]],
    method_summary: dict[str, dict[str, float]],
    figures_dir: Path,
) -> None:
    del grouped
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    labels, vals, bar_colors = _catastrophic_bwt_payload(method_summary)
    if vals.size == 0:
        plt.close(fig)
        return

    y_min = min(-0.25, float(vals.min()) - 0.03)
    y_max = max(0.22, float(vals.max()) + 0.04)
    ax.set_ylim(y_min, y_max)
    ax.axhspan(y_min, 0.0, color="#d62728", alpha=0.1, zorder=0)

    bars = ax.bar(labels, vals, color=bar_colors, zorder=2)
    ax.axhline(0.0, color="black", linewidth=2.0, zorder=3)
    ax.set_title("Catastrophic Forgetting — Backward Transfer (BWT)")
    ax.set_ylabel("BWT score")
    ax.set_xlabel("Method")
    ax.tick_params(axis="x", labelrotation=20)

    for bar, val in zip(bars, vals, strict=True):
        offset = 0.01 if val >= 0 else -0.015
        va = "bottom" if val >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(val) + offset,
            f"{float(val):+.3f}",
            ha="center",
            va=va,
            fontsize=9,
            color="#111111",
        )

    ax.text(0.02, 0.08, "Forgetting zone", transform=ax.transAxes, fontsize=9, color="#8b0000")
    ax.text(0.02, 0.92, "No forgetting", transform=ax.transAxes, fontsize=9, color="#1b5e20")
    fig.text(
        0.5,
        0.01,
        "Negative BWT = model forgets past tasks.\nPositive BWT = model improves on past tasks over time.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(figures_dir / "figure7_catastrophic_forgetting.png", dpi=180)
    plt.close(fig)


def _cleanup_legacy_figure_aliases(figures_dir: Path) -> None:
    # Remove stale aliases from earlier naming schemes to keep one figure per insight.
    legacy_files = [
        "catastrophic_forgetting.png",
        "claim5_stability.png",
        "figure3_per_task_accuracy_curve.png",
        "figure4_within_task_recovery_detail.png",
        "figure5_training_stability_seed_variance.png",
        "figure3_replay_ablation.png",
        "figure4_context_window_ablation.png",
    ]
    for name in legacy_files:
        (figures_dir / name).unlink(missing_ok=True)


def _write_sanity_table(
    *,
    method_summary: dict[str, dict[str, float]],
    out_dir: Path,
    ablation_results_path: str,
) -> None:
    ablation = {}
    if ablation_results_path:
        p = Path(ablation_results_path)
        if p.exists():
            ablation = json.loads(p.read_text(encoding="utf-8"))

    random_baseline = 0.10
    online = method_summary.get("online_sgd", {})
    er = method_summary.get("experience_replay", {})
    pi = method_summary.get("pi_transformer", {})
    e1 = float(ablation.get("replay_E_1", {}).get("rolling_accuracy", 0.0))
    e20 = float(ablation.get("replay_E_20", {}).get("rolling_accuracy", 0.0))

    rows = [
        ("Random baseline accuracy", "10.0%", f"{random_baseline:.4f}", "CHECK"),
        ("Online SGD shows forgetting", "BWT < 0", f"{online.get('bwt', float('nan')):.4f}" if online else "N/A", "PASS" if online and online.get("bwt", 1.0) < 0 else "FAIL"),
        ("E=1 near random", "~10%", f"{e1:.4f}" if ablation else "N/A", "PASS" if ablation and e1 <= 0.15 else ("N/A" if not ablation else "FAIL")),
        ("E=20 significantly better", ">30%", f"{e20:.4f}" if ablation else "N/A", "PASS" if ablation and e20 >= 0.30 else ("N/A" if not ablation else "FAIL")),
        ("Pi-Transformer > ER", "Yes", f"{pi.get('avg_acc_last_20_tasks', float('nan')):.4f} vs {er.get('avg_acc_last_20_tasks', float('nan')):.4f}" if pi and er else "N/A", "PASS" if pi and er and pi.get("avg_acc_last_20_tasks", 0.0) > er.get("avg_acc_last_20_tasks", 0.0) else "FAIL"),
        ("Label injection has effect", "diff > 0.05", "N/A", "N/A"),
        ("Cache persists across chunks", "Shape grows", "N/A", "N/A"),
    ]

    csv_path = out_dir / "sanity_check_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sanity Check", "Expected", "Actual", "Pass?"])
        writer.writerows(rows)

    md_path = out_dir / "sanity_check_table.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Sanity Check | Expected | Actual | Pass? |\n")
        f.write("|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |\n")


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    raw_dir = runs_dir / "raw"
    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(raw_dir)
    if not runs:
        raise SystemExit(f"No run files found in {raw_dir}")

    grouped = _group_by_method(runs)
    _plot_main_accuracy(grouped, figures_dir)
    method_summary = _write_comparison_tables(grouped, out_dir)
    _plot_per_task_curve(grouped, figures_dir)
    _plot_recovery_detail(grouped, figures_dir)
    _plot_seed_stability(grouped, figures_dir)
    _plot_catastrophic_forgetting(grouped, method_summary, figures_dir)
    _cleanup_legacy_figure_aliases(figures_dir)
    _write_sanity_table(method_summary=method_summary, out_dir=out_dir, ablation_results_path=args.ablation_results)
    print(f"Saved tables to {out_dir}")
    print(f"Saved figures to {figures_dir}")


if __name__ == "__main__":
    main()
