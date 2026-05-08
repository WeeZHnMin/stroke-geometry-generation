from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


STEP_RE = re.compile(
    r"epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)/(?P<total_steps>\d+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+acc=(?P<token_acc>[-+0-9.eE]+)\s+"
    r"x_acc=(?P<x_acc>[-+0-9.eE]+)\s+y_acc=(?P<y_acc>[-+0-9.eE]+)\s+pen_acc=(?P<pen_acc>[-+0-9.eE]+)\s+"
    r"xy_mae=(?P<xy_mae>[-+0-9.eE]+)"
)
EPOCH_RE = re.compile(
    r"epoch=(?P<epoch>\d+)\s+train_loss=(?P<train_loss>[-+0-9.eE]+)\s+"
    r"val_loss=(?P<val_loss>[-+0-9.eE]+)\s+val_acc=(?P<val_acc>[-+0-9.eE]+)\s+"
    r"val_x_acc=(?P<val_x_acc>[-+0-9.eE]+)\s+val_y_acc=(?P<val_y_acc>[-+0-9.eE]+)\s+"
    r"val_pen_acc=(?P<val_pen_acc>[-+0-9.eE]+)\s+val_xy_mae=(?P<val_xy_mae>[-+0-9.eE]+)"
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_text_log(path: Path) -> tuple[list[dict], list[dict]]:
    step_records: list[dict] = []
    epoch_records: list[dict] = []
    if not path.exists():
        return step_records, epoch_records

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        step_match = STEP_RE.search(line)
        if step_match:
            values = step_match.groupdict()
            step_records.append(
                {
                    "epoch": int(values["epoch"]),
                    "step": int(values["step"]),
                    "total_steps": int(values["total_steps"]),
                    "loss": float(values["loss"]),
                    "token_acc": float(values["token_acc"]),
                    "x_acc": float(values["x_acc"]),
                    "y_acc": float(values["y_acc"]),
                    "pen_acc": float(values["pen_acc"]),
                    "xy_mae": float(values["xy_mae"]),
                }
            )
            continue

        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            values = epoch_match.groupdict()
            epoch_records.append(
                {
                    "epoch": int(values["epoch"]),
                    "train": {"loss": float(values["train_loss"])},
                    "val": {
                        "loss": float(values["val_loss"]),
                        "token_acc": float(values["val_acc"]),
                        "x_acc": float(values["val_x_acc"]),
                        "y_acc": float(values["val_y_acc"]),
                        "pen_acc": float(values["val_pen_acc"]),
                        "xy_mae": float(values["val_xy_mae"]),
                    },
                }
            )
    return step_records, epoch_records


def step_x(records: Iterable[dict]) -> list[float]:
    xs = []
    for record in records:
        total = max(int(record.get("total_steps", 1)), 1)
        xs.append(float(record["epoch"]) - 1.0 + float(record["step"]) / total)
    return xs


def plot_metric(ax, epoch_records: list[dict], step_records: list[dict], metric: str, title: str) -> None:
    if step_records:
        xs = step_x(step_records)
        ys = [record[metric] for record in step_records if metric in record]
        xs = [x for x, record in zip(xs, step_records) if metric in record]
        if xs and ys:
            ax.plot(xs, ys, color="#8aa0b8", linewidth=1.0, alpha=0.65, label=f"train step {metric}")

    epochs = [record["epoch"] for record in epoch_records]
    train_values = [record.get("train", {}).get(metric) for record in epoch_records]
    val_values = [record.get("val", {}).get(metric) for record in epoch_records]

    train_points = [(x, y) for x, y in zip(epochs, train_values) if y is not None]
    val_points = [(x, y) for x, y in zip(epochs, val_values) if y is not None]
    if train_points:
        ax.plot(
            [x for x, _ in train_points],
            [y for _, y in train_points],
            marker="o",
            linewidth=2.0,
            label=f"train epoch {metric}",
        )
    if val_points:
        ax.plot(
            [x for x, _ in val_points],
            [y for _, y in val_points],
            marker="o",
            linewidth=2.0,
            label=f"val {metric}",
        )

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot action-token training metrics from a run directory or text log.")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory containing step_metrics.jsonl and epoch_metrics.jsonl.")
    parser.add_argument("--log-file", type=str, default=None, help="Plain text training log to parse when JSONL logs are absent.")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path. Defaults to <run-dir>/metrics.png.")
    args = parser.parse_args()

    if not args.run_dir and not args.log_file:
        raise SystemExit("Provide --run-dir or --log-file.")

    run_dir = Path(args.run_dir) if args.run_dir else None
    step_records: list[dict] = []
    epoch_records: list[dict] = []

    if run_dir:
        step_records = read_jsonl(run_dir / "step_metrics.jsonl")
        epoch_records = read_jsonl(run_dir / "epoch_metrics.jsonl")

    if (not step_records and not epoch_records) and args.log_file:
        step_records, epoch_records = parse_text_log(Path(args.log_file))

    if not step_records and not epoch_records:
        raise SystemExit("No metrics found. Expected JSONL logs or recognizable train console output.")

    output = Path(args.output) if args.output else (run_dir / "metrics.png" if run_dir else Path("metrics.png"))
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    plots = [
        ("loss", "Loss"),
        ("token_acc", "Token Accuracy"),
        ("x_acc", "X Accuracy"),
        ("y_acc", "Y Accuracy"),
        ("pen_acc", "Pen Accuracy"),
        ("xy_mae", "XY MAE"),
    ]
    for ax, (metric, title) in zip(axes.flat, plots):
        plot_metric(ax, epoch_records, step_records, metric, title)

    fig.suptitle(str(run_dir or args.log_file), fontsize=12)
    fig.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
