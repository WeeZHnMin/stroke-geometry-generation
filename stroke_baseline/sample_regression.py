import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR
from .regression_dataset import StrokeRegressionJsonlDataset
from .regression_model import CoordRegressionDecoderConfig, TextConditionedCoordRegressionModel

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_model(checkpoint_path: str | Path, device: torch.device, text_encoder_dir: str | None = None):
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
    cfg = CoordRegressionDecoderConfig(**state["decoder_cfg"])
    enc_dir = text_encoder_dir or state.get("text_encoder_dir", str(DEFAULT_TEXT_ENCODER_DIR))
    model = TextConditionedCoordRegressionModel(cfg, text_encoder_dir=enc_dir, max_text_len=state["max_text_len"])
    model.decoder.load_state_dict(state["decoder"])
    model.context_proj.load_state_dict(state["context_proj"])
    model.to(device)
    model.eval()
    return model, state


def denorm_xy(coords: torch.Tensor, canvas_size: float) -> torch.Tensor:
    xy = coords[..., :2].clone()
    return (xy + 1.0) * canvas_size / 2.0


def strokes_from_deltas(deltas: torch.Tensor) -> list[dict]:
    return [{"dx": float(dx), "dy": float(dy), "pen_state": "draw"} for dx, dy in deltas.tolist()]


def plot_teacher_forced(
    coords_abs: torch.Tensor,
    true_dxdy: torch.Tensor,
    pred_dxdy: torch.Tensor,
    mask: torch.Tensor,
    path: str | Path,
    title: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = mask.bool()
    coords_abs = coords_abs[valid].cpu()
    true_dxdy = true_dxdy[valid].cpu()
    pred_dxdy = pred_dxdy[valid].cpu()
    true_cum = torch.cat([coords_abs[:1], coords_abs[:1] + torch.cumsum(true_dxdy, dim=0)], dim=0)
    pred_cum = torch.cat([coords_abs[:1], coords_abs[:1] + torch.cumsum(pred_dxdy, dim=0)], dim=0)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    axes[0].set_title("segment prediction")
    for xy, true_d, pred_d in zip(coords_abs, true_dxdy, pred_dxdy):
        axes[0].plot([xy[0], xy[0] + true_d[0]], [xy[1], xy[1] + true_d[1]], color="0.70", linewidth=2)
        axes[0].plot([xy[0], xy[0] + pred_d[0]], [xy[1], xy[1] + pred_d[1]], color="#d62728", linewidth=1.4, alpha=0.75)

    axes[1].set_title("cumulative trajectory")
    axes[1].plot(true_cum[:, 0], true_cum[:, 1], color="0.25", linewidth=2, label="true")
    axes[1].plot(pred_cum[:, 0], pred_cum[:, 1], color="#d62728", linewidth=1.6, label="pred")
    axes[1].legend(loc="best", fontsize=8)

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        xs = torch.cat([true_cum[:, 0], pred_cum[:, 0], coords_abs[:, 0]])
        ys = torch.cat([true_cum[:, 1], pred_cum[:, 1], coords_abs[:, 1]])
        span = float(max((xs.max() - xs.min()).item(), (ys.max() - ys.min()).item(), 1.0))
        pad = span * 0.08
        ax.set_xlim(float(xs.min()) - pad, float(xs.max()) + pad)
        ax.set_ylim(float(ys.max()) + pad, float(ys.min()) - pad)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_autoreg(points: torch.Tensor, path: str | Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.plot(points[:, 0].cpu(), points[:, 1].cpu(), color="black", linewidth=2)
    ax.scatter(points[:1, 0].cpu(), points[:1, 1].cpu(), s=24, color="#2ca02c", label="start")
    ax.scatter(points[-1:, 0].cpu(), points[-1:, 1].cpu(), s=24, color="#d62728", label="end")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    span = float(max((points[:, 0].max() - points[:, 0].min()).item(), (points[:, 1].max() - points[:, 1].min()).item(), 1.0))
    pad = span * 0.08
    ax.set_xlim(float(points[:, 0].min()) - pad, float(points[:, 0].max()) + pad)
    ax.set_ylim(float(points[:, 1].max()) + pad, float(points[:, 1].min()) - pad)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@torch.no_grad()
def teacher_forced_sample(args, model, state, device):
    canvas_size = float(args.canvas_size or state.get("canvas_size", 8.0))
    dataset = StrokeRegressionJsonlDataset(
        args.data,
        max_len=model.decoder.cfg.max_len,
        canvas_size=canvas_size,
        max_delta=model.decoder.cfg.max_delta,
        draw_only=bool(state.get("draw_only", True)),
    )
    sample = dataset[args.sample_index]
    batch_coords = sample["decoder_coords"].unsqueeze(0).to(device)
    batch_mask = sample["target_mask"].unsqueeze(0).to(device)
    out = model(prompts=[sample["prompt"]], decoder_coords=batch_coords, target_mask=batch_mask)
    pred = out["pred_dxdy"][0].cpu()
    coords_abs = denorm_xy(sample["decoder_coords"], canvas_size)
    target = sample["target_dxdy"]
    mask = sample["target_mask"]
    plot_teacher_forced(coords_abs, target, pred, mask, args.png, sample["prompt"])
    payload = {
        "mode": "teacher_forced",
        "prompt": sample["prompt"],
        "sample_index": args.sample_index,
        "target_strokes": strokes_from_deltas(target[mask]),
        "pred_strokes": strokes_from_deltas(pred[mask]),
    }
    return payload


@torch.no_grad()
def autoreg_sample(args, model, state, device):
    canvas_size = float(args.canvas_size or state.get("canvas_size", 8.0))
    x = float(args.start_x)
    y = float(args.start_y)
    points = [[x, y]]
    deltas = []
    for i in range(args.max_steps):
        coord = torch.tensor(
            [[[x / canvas_size * 2.0 - 1.0, y / canvas_size * 2.0 - 1.0, i / max(model.decoder.cfg.max_len - 1, 1)]]],
            dtype=torch.float32,
            device=device,
        )
        mask = torch.ones(1, 1, dtype=torch.bool, device=device)
        out = model(prompts=[args.prompt], decoder_coords=coord, target_mask=mask)
        dx, dy = out["pred_dxdy"][0, 0].cpu().tolist()
        deltas.append([dx, dy])
        x += dx
        y += dy
        points.append([x, y])
    points_tensor = torch.tensor(points, dtype=torch.float32)
    plot_autoreg(points_tensor, args.png, args.prompt)
    return {
        "mode": "autoreg",
        "prompt": args.prompt,
        "start": [args.start_x, args.start_y],
        "strokes": [{"dx": float(dx), "dy": float(dy), "pen_state": "draw"} for dx, dy in deltas],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize continuous regression decoder predictions.")
    parser.add_argument("--checkpoint", type=str, default="runs/stroke_regression_scale8/checkpoint.pt")
    parser.add_argument("--mode", choices=["teacher", "autoreg"], default="teacher")
    parser.add_argument("--data", type=str, default="generated_data/polar_balanced/polar_balanced_scale8_train.jsonl")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="一个三角形")
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--canvas-size", type=float, default=None)
    parser.add_argument("--png", type=str, default="viz_output/regression_sample.png")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--text-encoder-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state = load_model(args.checkpoint, device, text_encoder_dir=args.text_encoder_dir)
    if args.mode == "teacher":
        payload = teacher_forced_sample(args, model, state, device)
    else:
        payload = autoreg_sample(args, model, state, device)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {args.png}")


if __name__ == "__main__":
    main()
