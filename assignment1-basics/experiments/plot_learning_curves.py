"""Plot TinyStories / ablation / OWT learning curves for the writeup.

Reads experiments/artifacts/**/ckpt.jsonl and writes PNGs to
notes/assignment1/figures/.

Usage (from assignment1-basics):
  uv run python experiments/plot_learning_curves.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
FIGURES_DIR = ASSIGNMENT_DIR.parent / "notes" / "assignment1" / "figures"
CONTEXT_LENGTH = 256


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def series(rows: list[dict], y_key: str, x_key: str = "step") -> tuple[list, list]:
    xs, ys = [], []
    for row in rows:
        y = row.get(y_key)
        if y is None:
            continue
        xs.append(row[x_key])
        ys.append(y)
    return xs, ys


def tokens_x(rows: list[dict], batch_size: int, y_key: str) -> tuple[list, list]:
    xs, ys = [], []
    for row in rows:
        y = row.get(y_key)
        if y is None:
            continue
        xs.append(row["step"] * batch_size * CONTEXT_LENGTH)
        ys.append(y)
    return xs, ys


def save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_train_valid(rows: list[dict], title: str, filename: str, ylim=None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, x_key, xlabel in (
        (axes[0], "step", "step"),
        (axes[1], "wall_time_seconds", "wall clock (s)"),
    ):
        ax.plot(*series(rows, "train_loss", x_key), color="C0", alpha=0.45, linewidth=1.0, label="train")
        ax.plot(
            *series(rows, "valid_loss", x_key), color="C0", marker="o", markersize=3.5, linewidth=1.5, label="valid"
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend()
    save(fig, filename)


def plot_overlay(
    runs: list[tuple[str, list[dict], str]],
    y_key: str,
    title: str,
    filename: str,
    *,
    ylim=None,
    x_mode: str = "step",
    batch_sizes: dict[str, int] | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, x_kind, xlabel in (
        (axes[0], x_mode, "tokens" if x_mode == "tokens" else "step"),
        (axes[1], "wall", "wall clock (s)"),
    ):
        for label, rows, color in runs:
            if x_kind == "tokens":
                xs, ys = tokens_x(rows, batch_sizes[label], y_key)
            elif x_kind == "wall":
                xs, ys = series(rows, y_key, "wall_time_seconds")
            else:
                xs, ys = series(rows, y_key, "step")
            ax.plot(xs, ys, color=color, marker="o", markersize=3.0, linewidth=1.5, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(y_key.replace("_", " "))
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend()
    save(fig, filename)


def main() -> None:
    ts = load_jsonl(ARTIFACTS / "tinystories_lm" / "ckpt.jsonl")
    owt = load_jsonl(ARTIFACTS / "owt_lm" / "ckpt.jsonl")
    lr_runs = [
        ("1e-4", load_jsonl(ARTIFACTS / "sweeps" / "lr_1e-4" / "ckpt.jsonl"), "C3"),
        ("3e-4", load_jsonl(ARTIFACTS / "sweeps" / "lr_3e-4" / "ckpt.jsonl"), "C2"),
        ("1e-3", ts, "C0"),
        ("3e-3", load_jsonl(ARTIFACTS / "sweeps" / "lr_3e-3" / "ckpt.jsonl"), "C1"),
        ("1e-2", load_jsonl(ARTIFACTS / "sweeps" / "lr_1e-2" / "ckpt.jsonl"), "C4"),
        ("1e-1", load_jsonl(ARTIFACTS / "sweeps" / "lr_1e-1" / "ckpt.jsonl"), "C5"),
    ]
    batch_runs = [
        ("16", load_jsonl(ARTIFACTS / "sweeps" / "batch_16" / "ckpt.jsonl"), "C3"),
        ("32", ts, "C0"),
        ("64", load_jsonl(ARTIFACTS / "sweeps" / "batch_64" / "ckpt.jsonl"), "C2"),
        ("128", load_jsonl(ARTIFACTS / "sweeps" / "batch_128" / "ckpt.jsonl"), "C1"),
    ]
    no_rms = load_jsonl(ARTIFACTS / "ablations" / "no_rms" / "ckpt.jsonl")
    no_rms_low = load_jsonl(ARTIFACTS / "ablations" / "no_rms_1e-4" / "ckpt.jsonl")
    post_norm = load_jsonl(ARTIFACTS / "ablations" / "post_norm" / "ckpt.jsonl")
    nope = load_jsonl(ARTIFACTS / "ablations" / "nope" / "ckpt.jsonl")
    silu = load_jsonl(ARTIFACTS / "ablations" / "silu" / "ckpt.jsonl")
    lr_1e4 = lr_runs[0][1]

    plot_train_valid(ts, "TinyStories baseline", "ts_main.png", ylim=(1.2, 4.0))
    plot_overlay(lr_runs, "valid_loss", "TinyStories learning-rate sweep", "ts_lr.png")
    plot_overlay(
        batch_runs,
        "valid_loss",
        "TinyStories batch-size sweep",
        "ts_batch.png",
        x_mode="tokens",
        batch_sizes={"16": 16, "32": 32, "64": 64, "128": 128},
    )

    def after(rows, min_step):
        return [r for r in rows if r["step"] >= min_step]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].plot(*series(ts, "valid_loss"), color="C0", marker="o", markersize=3.0, linewidth=1.5, label="baseline 1e-3")
    axes[0].plot(*series(no_rms, "valid_loss"), color="C3", marker="o", markersize=3.0, linewidth=1.5, label="no RMSNorm 1e-3")
    axes[0].set_yscale("log")
    axes[0].set_title("Remove RMSNorm (original LR)")
    axes[1].plot(*series(ts, "valid_loss"), color="C0", marker="o", markersize=3.0, linewidth=1.5, label="baseline 1e-3")
    axes[1].plot(*series(lr_1e4, "valid_loss"), color="C2", marker="o", markersize=3.0, linewidth=1.5, label="baseline 1e-4")
    axes[1].plot(*series(after(no_rms_low, 5000), "valid_loss"), color="C1", marker="o", markersize=3.0, linewidth=1.5, label="no RMSNorm 1e-4")
    axes[1].set_ylim(1.2, 8.8)
    axes[1].set_title("Remove RMSNorm (lower LR, from 5k)")
    for ax in axes:
        ax.set_xlabel("step")
        ax.set_ylabel("valid loss")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()
    save(fig, "ablation_rmsnorm.png")
    plot_overlay(
        [("pre-norm (baseline)", ts, "C0"), ("post-norm", post_norm, "C1")],
        "valid_loss",
        "Pre-norm vs post-norm",
        "ablation_post_norm.png",
    )
    plot_overlay(
        [("RoPE (baseline)", ts, "C0"), ("NoPE", nope, "C1")],
        "valid_loss",
        "RoPE vs NoPE",
        "ablation_nope.png",
    )
    plot_overlay(
        [("SwiGLU (baseline)", ts, "C0"), ("SiLU", silu, "C1")],
        "valid_loss",
        "SwiGLU vs SiLU",
        "ablation_silu.png",
    )
    plot_train_valid(owt, "OpenWebText", "owt_main.png", ylim=(3.8, 6.5))
    plot_leaderboard()
    plot_iters10k()


def plot_leaderboard() -> None:
    runs = [
        ("baseline", ARTIFACTS / "leaderboard" / "owt_ctx512_bf16" / "ckpt.jsonl"),
        ("QKNorm", ARTIFACTS / "leaderboard" / "owt_ctx512_qknorm_bf16" / "ckpt.jsonl"),
        ("tying", ARTIFACTS / "leaderboard" / "owt_ctx512_weight_tying_bf16" / "ckpt.jsonl"),
        ("Muon", ARTIFACTS / "leaderboard" / "owt_ctx512_muon_bf16" / "ckpt.jsonl"),
        ("QKNorm+tying", ARTIFACTS / "leaderboard" / "owt_ctx512_qknorm+tying_bf16" / "ckpt.jsonl"),
        ("QKNorm+tying+Muon", ARTIFACTS / "leaderboard" / "owt_ctx512_qknorm+tying+muon_bf16" / "ckpt.jsonl"),
        ("QKNorm+tying L6", ARTIFACTS / "leaderboard" / "owt_ctx512_qknorm+tying_l6_bf16" / "ckpt.jsonl"),
    ]
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    for i, (label, path) in enumerate(runs):
        if not path.is_file():
            continue
        rows = load_jsonl(path)
        color = f"C{i}"
        ax.plot(*series(rows, "train_loss"), color=color, alpha=0.35, linewidth=1.0)
        ax.plot(
            *series(rows, "valid_loss"),
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            label=label,
        )
    ax.set_xlabel("iters")
    ax.set_ylabel("loss")
    ax.set_title("OWT leaderboard: train (thin) and valid (markers)")
    ax.set_ylim(3.5, 6.6)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    save(fig, "owt_leaderboard.png")


def plot_iters10k() -> None:
    root = ARTIFACTS / "iters10k"
    runs = [
        ("baseline", "owt_ctx512_bf16", "C0"),
        ("Muon", "owt_ctx512_muon_bf16", "C1"),
        ("QKNorm", "owt_ctx512_qknorm_bf16", "C2"),
        ("tying", "owt_ctx512_weight_tying_bf16", "C3"),
        ("zero-init", "owt_ctx512_zeroinit_bf16", "C4"),
        ("QKNorm+tying", "owt_ctx512_qknorm+tying_bf16", "C5"),
        ("QKNorm+tying L6", "owt_ctx512_qknorm+tying_l6_bf16", "C6"),
        ("QKNorm+tying+Muon", "owt_ctx512_qknorm+tying+muon_bf16", "C7"),
        ("final (+zero-init)", "owt_ctx512_final_bf16", "C8"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6))
    finals = []
    for label, stem, color in runs:
        path = root / stem / "ckpt.jsonl"
        if not path.is_file():
            continue
        rows = load_jsonl(path)
        axes[0].plot(
            *series(rows, "valid_loss"),
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            label=label,
        )
        axes[1].plot(
            *series(rows, "valid_loss", "wall_time_seconds"),
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            label=label,
        )
        last_valid = next((r["valid_loss"] for r in reversed(rows) if "valid_loss" in r), None)
        last = rows[-1]
        finals.append((label, last_valid, last["wall_time_seconds"]))

    axes[0].set_xlabel("step")
    axes[1].set_xlabel("wall clock (s)")
    for ax in axes:
        ax.set_ylabel("valid loss")
        ax.set_ylim(3.80, 6.55)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title("OWT 10k sweep: valid vs step")
    axes[1].set_title("OWT 10k sweep: valid vs wall clock")
    save(fig, "owt_iters10k.png")

    print("iters10k final valid_loss")
    for label, valid, wall in sorted(finals, key=lambda x: x[1] if x[1] is not None else 9e9):
        print(f"  {valid:.4f}  {wall:6.0f}s  {label}")


def plot_final_run() -> None:
    path = ARTIFACTS / "leaderboard" / "final" / "ckpt.jsonl"
    rows = load_jsonl(path)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    axes[0].plot(*series(rows, "train_loss"), color="C0", alpha=0.35, linewidth=1.0, label="train")
    axes[0].plot(
        *series(rows, "valid_loss"),
        color="C0",
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        label="valid",
    )
    axes[0].axvline(1500, color="0.6", linestyle="--", linewidth=1.0)
    axes[0].axvline(42500, color="0.6", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].set_ylim(3.35, 5.2)
    axes[0].set_title("OWT final: WSD L4 + Muon")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(*series(rows, "learning_rate"), color="C1", linewidth=1.6, label="AdamW lr")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("learning rate")
    axes[1].set_title("WSD schedule")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    save(fig, "owt_final.png")


def plot_cosine_vs_wsd() -> None:
    wsd = load_jsonl(ARTIFACTS / "leaderboard" / "final" / "ckpt.jsonl")
    cosine = load_jsonl(ARTIFACTS / "leaderboard" / "owt_ctx512_cosine_bf16" / "ckpt.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for rows, color, label in ((wsd, "C0", "WSD"), (cosine, "C1", "cosine")):
        axes[0].plot(*series(rows, "train_loss"), color=color, alpha=0.30, linewidth=1.0)
        axes[0].plot(
            *series(rows, "valid_loss"),
            color=color,
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            label=label,
        )
    axes[0].axvline(1500, color="0.6", linestyle="--", linewidth=1.0)
    axes[0].axvline(42500, color="0.6", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].set_ylim(3.35, 5.2)
    axes[0].set_title("valid (markers) + train (thin)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(*series(wsd, "learning_rate"), color="C0", linewidth=1.6, label="WSD")
    axes[1].plot(*series(cosine, "learning_rate"), color="C1", linewidth=1.6, label="cosine")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("AdamW lr")
    axes[1].set_title("schedule")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    save(fig, "owt_cosine_vs_wsd.png")


if __name__ == "__main__":
    main()
