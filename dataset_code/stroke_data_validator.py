from pathlib import Path

from stroke_data_factory.validator import validate_jsonl


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate generated stroke dataset jsonl files.")
    parser.add_argument("path", type=str, help="Path to jsonl dataset file")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--show", type=int, default=10, help="How many issues to print")
    args = parser.parse_args()

    result = validate_jsonl(Path(args.path), max_samples=args.max_samples)
    print(f"path: {result['path']}")
    print(f"validated_samples: {result['validated_samples']}")
    print(f"issue_count: {result['issue_count']}")
    print(f"issue_summary: {result['issue_summary']}")

    for issue in result["issues"][: args.show]:
        print(f"[sample {issue.sample_index}] {issue.code}: {issue.message}")


if __name__ == "__main__":
    main()
