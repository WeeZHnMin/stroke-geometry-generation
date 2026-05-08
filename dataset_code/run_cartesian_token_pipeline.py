from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_python_module(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *args]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot pipeline: generate continuous raw data, export compact Cartesian vocab, optionally train CPCF.")
    parser.add_argument("--raw-output", type=str, default="generated_data/continuous_geometry/continuous_geometry_train.jsonl")
    parser.add_argument("--vocab-output", type=str, default="generated_data/cartesian_tokens/cartesian_compact_vocab.json")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--samples-per-combo", type=int, default=56)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canvas-size", type=float, default=8.0)
    parser.add_argument("--sample-step", type=float, default=0.1)
    parser.add_argument("--move-step", type=float, default=0.5)
    parser.add_argument("--max-delta", type=float, default=0.5)
    parser.add_argument("--max-action-len", type=int, default=192)
    parser.add_argument("--rotate", action="store_true")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train-output-dir", type=str, default="runs/stroke_action_cpcf_compact")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--xy-hidden-dim", type=int, default=128)
    parser.add_argument("--pen-emb-dim", type=int, default=32)
    parser.add_argument("--input-kernel-size", type=int, default=3)
    args = parser.parse_args()

    raw_output = Path(args.raw_output)
    if args.generate:
        gen_args = [
            "dataset_code/generate_continuous_geometry_dataset.py",
            "--samples-per-combo",
            str(args.samples_per_combo),
            "--seed",
            str(args.seed),
            "--canvas-size",
            str(args.canvas_size),
            "--sample-step",
            str(args.sample_step),
            "--move-step",
            str(args.move_step),
            "--max-delta",
            str(args.max_delta),
            "--max-action-len",
            str(args.max_action_len),
            "--output",
            str(raw_output),
        ]
        if args.rotate:
            gen_args.append("--rotate")
        subprocess.run([sys.executable, *gen_args], check=True)

    run_python_module(
        "dataset_code.export_cartesian_vocab",
        ["--data", str(raw_output), "--output", args.vocab_output],
    )

    if args.train:
        train_args = [
            "--data",
            str(raw_output),
            "--vocab-file",
            args.vocab_output,
            "--output-dir",
            args.train_output_dir,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--max-action-len",
            str(args.max_action_len),
            "--max-text-len",
            str(args.max_text_len),
            "--d-model",
            str(args.d_model),
            "--n-heads",
            str(args.n_heads),
            "--decoder-layers",
            str(args.decoder_layers),
            "--dropout",
            str(args.dropout),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--grad-clip",
            str(args.grad_clip),
            "--val-ratio",
            str(args.val_ratio),
            "--log-every",
            str(args.log_every),
            "--attention-variant",
            "hetero",
            "--input-mode",
            "cpcf",
            "--xy-hidden-dim",
            str(args.xy_hidden_dim),
            "--pen-emb-dim",
            str(args.pen_emb_dim),
            "--input-kernel-size",
            str(args.input_kernel_size),
        ]
        run_python_module("stroke_baseline.train_action_tokens", train_args)


if __name__ == "__main__":
    main()
