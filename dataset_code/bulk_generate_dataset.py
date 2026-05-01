import json
from datetime import datetime
from pathlib import Path

from stroke_data_factory.generator import generate_dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
BULK_DIR = ROOT_DIR / "generated_data" / "bulk"
MANIFEST_DIR = ROOT_DIR / "generated_data" / "manifests"


DEFAULT_PLAN = [
    ("foundation_shapes_v3_easy", "balanced_v1", "easy", 20000),
    ("foundation_shapes_v3_mixed", "balanced_v1", "medium", 20000),
    ("foundation_shapes_v3_hard", "balanced_v1", "hard", 10000),
    ("coordinate_grounding", "coordinate_focus", "easy", 20000),
    ("spatial_instruction", "spatial_instruction_focus", None, 20000),
    ("spatial_attribute", "spatial_attribute_focus", None, 20000),
    ("spatial_relation_attribute", "spatial_relation_focus", None, 20000),
    ("contrast_pair", "contrast_pair_focus", None, 20000),
    ("equation_function", "equation_focus", None, 20000),
    ("advanced_composition", "advanced_composition_focus", None, 20000),
    ("counting_group", "counting_group_focus", None, 20000),
    ("alignment_group", "alignment_group_focus", None, 20000),
    ("edge_contact", "edge_contact_focus", None, 20000),
    ("relative_size", "relative_size_focus", None, 20000),
    ("same_diff", "same_diff_focus", None, 20000),
    ("through_points", "through_points_focus", None, 20000),
    ("region_occupancy", "region_occupancy_focus", None, 20000),
    ("step_instruction", "step_instruction_focus", None, 20000),
    ("negation_constraint", "negation_constraint_focus", None, 20000),
    ("style_precision", "style_precision_focus", None, 20000),
    ("logic_conjunction", "logic_conjunction_focus", None, 20000),
    ("reference_layout", "reference_layout_focus", None, 20000),
]


def _save_jsonl(samples, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Bulk-generate large geometric stroke datasets.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0, help="Multiply default sample counts by this value.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N plan entries.")
    parser.add_argument("--prefix", type=str, default="bulk")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BULK_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    plan = DEFAULT_PLAN[: args.limit] if args.limit is not None else DEFAULT_PLAN
    manifest = []

    for idx, (name, recipe, difficulty, count) in enumerate(plan):
        scaled_count = max(1, int(count * args.scale))
        dataset_seed = args.seed + idx
        samples = generate_dataset(
            num_samples=scaled_count,
            seed=dataset_seed,
            recipe_name=recipe,
            difficulty=difficulty,
        )
        filename = f"{args.prefix}_{name}_{timestamp}.jsonl"
        output_path = BULK_DIR / filename
        _save_jsonl(samples, output_path)
        manifest.append(
            {
                "name": name,
                "recipe": recipe,
                "difficulty": difficulty,
                "num_samples": scaled_count,
                "seed": dataset_seed,
                "path": str(output_path),
            }
        )
        print(f"wrote {scaled_count} samples to {output_path}")

    manifest_path = MANIFEST_DIR / f"{args.prefix}_manifest_{timestamp}.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "prefix": args.prefix,
                "scale": args.scale,
                "entries": manifest,
                "total_samples": sum(entry["num_samples"] for entry in manifest),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
