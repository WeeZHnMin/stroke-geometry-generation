import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import StrokeJsonlDataset
from .model import StrokeTransformerBaseline, StrokeTransformerConfig
from .sample import generate_strokes
from .tokenizer import CharTokenizer, HFTokenizer
from .visualize import save_strokes_png


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(batch: dict, out: dict) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["target_mask"]
    pred_dxdy = out["pred_dxdy"][mask]
    target_dxdy = batch["target_dxdy"][mask]
    pred_pen = out["pred_pen_logits"].reshape(-1, out["pred_pen_logits"].size(-1))
    target_pen = batch["target_pen"].reshape(-1)

    delta_loss = F.mse_loss(pred_dxdy, target_dxdy)
    pen_loss = F.cross_entropy(pred_pen, target_pen, ignore_index=-100)
    loss = delta_loss + pen_loss
    return loss, {"delta_loss": float(delta_loss.item()), "pen_loss": float(pen_loss.item())}


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def save_checkpoint(
    output_dir: Path,
    model: StrokeTransformerBaseline,
    dataset: StrokeJsonlDataset,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, output_dir / "model.pt")
    if isinstance(dataset.tokenizer, HFTokenizer):
        dataset.tokenizer.save(output_dir / "tokenizer")
    else:
        dataset.tokenizer.save(output_dir / "tokenizer.json")
    (output_dir / "config.json").write_text(json.dumps(model.cfg.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal text-to-stroke Transformer baseline.")
    parser.add_argument("--data", type=str, default="generated_data/raw/expanded_shapes_easy_20260403_103355.jsonl")
    parser.add_argument("--output-dir", type=str, default="runs/stroke_baseline")
    parser.add_argument("--tokenizer-type", choices=["hf", "char"], default="hf")
    parser.add_argument("--tokenizer-name", type=str, default="google-bert/bert-base-chinese")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-text-len", type=int, default=96)
    parser.add_argument("--max-stroke-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = None
    if args.tokenizer_type == "hf":
        tokenizer = HFTokenizer(args.tokenizer_name)
    elif args.tokenizer_type == "char":
        tokenizer = None

    dataset = StrokeJsonlDataset(
        args.data,
        tokenizer=tokenizer,
        max_text_len=args.max_text_len,
        max_stroke_len=args.max_stroke_len,
        limit=args.limit,
    )
    if args.tokenizer_type == "char":
        dataset.tokenizer = CharTokenizer.build(sample.prompt for sample in dataset.samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    cfg = StrokeTransformerConfig(
        vocab_size=dataset.tokenizer.vocab_size,
        max_text_len=args.max_text_len,
        max_stroke_len=args.max_stroke_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dropout=args.dropout,
    )
    model = StrokeTransformerBaseline(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"device={device} samples={len(dataset)} vocab={dataset.tokenizer.vocab_size}")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(
                batch["text_ids"],
                batch["text_mask"],
                batch["decoder_dxdy"],
                batch["decoder_pen"],
                target_mask=batch["target_mask"],
            )
            loss, parts = compute_loss(batch, out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            running += loss.item()
            if global_step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={global_step} loss={loss.item():.4f} "
                    f"delta={parts['delta_loss']:.4f} pen={parts['pen_loss']:.4f}"
                )

        avg = running / max(len(loader), 1)
        print(f"epoch={epoch} avg_loss={avg:.4f}")

    output_dir = Path(args.output_dir)
    save_checkpoint(output_dir, model, dataset, args)

    prompt = dataset.samples[0].prompt
    model.eval()
    strokes = generate_strokes(model, dataset.tokenizer, prompt, max_steps=64, device=device)
    sample_path = output_dir / "sample.png"
    save_strokes_png(strokes, sample_path, title=prompt)
    print(f"saved checkpoint to {output_dir}")
    print(f"sample prompt: {prompt}")
    print(f"sample png: {sample_path}")


if __name__ == "__main__":
    main()
