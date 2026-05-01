import json
from datetime import datetime
from pathlib import Path

from stroke_data_factory.generator import generate_dataset
from stroke_data_factory.spec_sampler import RECIPE_PRESETS


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "generated_data"
DEFAULT_RAW_DIR = DEFAULT_OUTPUT_DIR / "raw"


def build_default_output_path(prefix: str = "toy_stroke_dataset") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RAW_DIR / f"{prefix}_{timestamp}.jsonl"


def save_jsonl(samples, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate a toy geometric stroke dataset.")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--name", type=str, default="toy_stroke_dataset")
    parser.add_argument("--recipe", type=str, default="balanced_v1", choices=sorted(RECIPE_PRESETS.keys()))
    parser.add_argument("--difficulty", type=str, default=None, choices=["easy", "medium", "hard"])
    args = parser.parse_args()

    samples = generate_dataset(
        args.num_samples,
        args.seed,
        recipe_name=args.recipe,
        difficulty=args.difficulty,
    )
    output_path = Path(args.output) if args.output else build_default_output_path(args.name)
    save_jsonl(samples, output_path)

    print(f"wrote {len(samples)} samples to {output_path}")
    print("recipe:", args.recipe, "difficulty:", args.difficulty)
    print("example prompt:", samples[0]["prompt"])
    print("example metadata:", samples[0]["metadata"])
    print("example strokes:", samples[0]["strokes"][:4], "...")


if __name__ == "__main__":
    main()
