from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stroke_baseline.dataset import read_jsonl
from stroke_baseline.pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR
from stroke_baseline.visualize import save_strokes_png

from .dataset import PEN_STATE_TO_ID
from .model import STEP_DIM, StrokeDiffusionConfig, TextConditionedStrokeDiffusionModel
from .scheduler import DiffusionScheduler


PEN_ID_TO_STATE = {pen_id: pen_state for pen_state, pen_id in PEN_STATE_TO_ID.items()}


def load_model(checkpoint: str | Path, device: torch.device) -> tuple[TextConditionedStrokeDiffusionModel, dict]:
    state = torch.load(checkpoint, map_location=device)
    cfg = StrokeDiffusionConfig(**state["cfg"])
    text_encoder_dir = Path(state.get("text_encoder_dir", DEFAULT_TEXT_ENCODER_DIR))
    if not text_encoder_dir.exists():
        text_encoder_dir = Path(DEFAULT_TEXT_ENCODER_DIR)
    model = TextConditionedStrokeDiffusionModel(cfg, text_encoder_dir=text_encoder_dir).to(device)
    model.denoiser.load_state_dict(state["denoiser"])
    model.context_proj.load_state_dict(state["context_proj"])
    model.eval()
    return model, state


def predict_x0(
    model: TextConditionedStrokeDiffusionModel,
    xt: torch.Tensor,
    timestep: int,
    prompt: str,
    seq_mask: torch.Tensor,
) -> torch.Tensor:
    t = torch.full((xt.size(0),), timestep, device=xt.device, dtype=torch.long)
    out = model(
        prompts=[prompt] * xt.size(0),
        noisy_steps=xt,
        timesteps=t,
        seq_mask=seq_mask,
    )
    return out["pred_steps"]


@torch.no_grad()
def sample_sequence(
    model: TextConditionedStrokeDiffusionModel,
    *,
    prompt: str,
    max_steps: int,
    canvas_size: float,
    num_train_timesteps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    if seed:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    scheduler = DiffusionScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
    ).to(device)

    xt = torch.randn(1, max_steps, STEP_DIM, device=device)
    seq_mask = torch.ones(1, max_steps, device=device, dtype=torch.bool)

    for timestep in reversed(range(num_train_timesteps)):
        pred_x0 = predict_x0(model, xt, timestep, prompt, seq_mask)
        pred_x0 = pred_x0.clone()
        pred_x0[..., :2] = pred_x0[..., :2].clamp(0.0, canvas_size)
        pred_x0[..., 2] = pred_x0[..., 2].clamp(0.0, 2.0)

        alpha_t = scheduler.alphas[timestep]
        alpha_bar_t = scheduler.alpha_bars[timestep]
        beta_t = scheduler.betas[timestep]
        sqrt_one_minus_alpha_bar_t = scheduler.sqrt_one_minus_alpha_bars[timestep]

        eps_pred = (xt - torch.sqrt(alpha_bar_t) * pred_x0) / sqrt_one_minus_alpha_bar_t.clamp_min(1e-8)
        mean = (xt - (beta_t / sqrt_one_minus_alpha_bar_t.clamp_min(1e-8)) * eps_pred) / torch.sqrt(alpha_t)

        if timestep > 0:
            noise = torch.randn_like(xt)
            xt = mean + torch.sqrt(beta_t) * noise
        else:
            xt = pred_x0

    return xt[0].detach().cpu()


def absolute_steps_to_sequences(steps: torch.Tensor) -> tuple[list[dict], list[dict]]:
    pen_ids = steps[:, 2].round().clamp(min=0, max=2).long().tolist()
    points = steps[:, :2].tolist()

    end_idx = None
    for idx, pen_id in enumerate(pen_ids):
        if pen_id == 2:
            end_idx = idx
            break
    if end_idx is None:
        end_idx = len(pen_ids) - 1

    absolute_steps: list[dict] = []
    strokes: list[dict] = []
    prev_x = 0.0
    prev_y = 0.0
    for idx in range(end_idx + 1):
        x, y = points[idx]
        pen_id = 2 if idx == end_idx else pen_ids[idx]
        pen_state = PEN_ID_TO_STATE[pen_id]
        absolute_steps.append(
            {
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "pen_id": int(pen_id),
                "pen_state": pen_state,
            }
        )
        strokes.append(
            {
                "dx": round(float(x - prev_x), 4),
                "dy": round(float(y - prev_y), 4),
                "pen_state": pen_state,
            }
        )
        prev_x = float(x)
        prev_y = float(y)
    return absolute_steps, strokes


def save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_from_dataset(path: str | Path, sample_index: int) -> dict:
    samples = read_jsonl(path)
    return samples[sample_index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from trained stroke diffusion checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--canvas-size", type=float, default=6.0)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=str, default="viz_output/diffusion_sample.json")
    parser.add_argument("--png", type=str, default="viz_output/diffusion_sample.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _state = load_model(args.checkpoint, device)

    prompt = args.prompt
    if prompt is None and args.data:
        raw = sample_from_dataset(args.data, args.sample_index)
        prompt = str(raw["prompt"])
    if prompt is None:
        raise ValueError("Provide either --prompt or --data with --sample-index.")

    steps = sample_sequence(
        model,
        prompt=prompt,
        max_steps=min(args.max_steps, model.cfg.max_seq_len),
        canvas_size=args.canvas_size,
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
        seed=args.seed,
    )
    absolute_steps, strokes = absolute_steps_to_sequences(steps)

    payload = {
        "prompt": prompt,
        "num_steps": len(absolute_steps),
        "absolute_sequence": absolute_steps,
        "strokes": strokes,
    }
    save_json(payload, args.json)
    save_strokes_png(strokes, args.png, title=prompt)

    print(f"device={device}")
    print(f"prompt={prompt}")
    print(f"num_steps={len(absolute_steps)}")
    print(f"saved_json={args.json}")
    print(f"saved_png={args.png}")


if __name__ == "__main__":
    main()
