import json

with open(r"C:\Users\34619\Desktop\Program\stroke-geometry-generation\generated_data\chinese_mvp\chinese_mvp_12k.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()[:5]

for i, line in enumerate(lines):
    s = json.loads(line)
    print(f"=== sample {i} ===")
    print("prompt:", s["prompt"])
    print("shape:", s["shapes"][0]["shape_type"])
    print("bbox:", s["shapes"][0]["bbox"])
    print("strokes:", s["strokes"])
    print()
