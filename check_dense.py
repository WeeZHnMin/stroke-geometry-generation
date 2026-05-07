import json
path = r"C:\Users\34619\Desktop\Program\stroke-geometry-generation\generated_data\chinese_mvp\chinese_mvp_dense_8k.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        s = json.loads(line)
        st = s["shapes"][0]["shape_type"]
        n = len(s["strokes"])
        print(f"{st:20s} -> {n:3d} steps ({n*3:3d} tokens)")
