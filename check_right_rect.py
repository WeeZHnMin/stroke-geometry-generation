import json
path = r"C:\Users\34619\Desktop\Program\stroke-geometry-generation\generated_data\chinese_mvp\chinese_mvp_dense_8k.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        s = json.loads(line)
        if s["shapes"][0]["shape_type"] == "rectangle" and "右边" in s["prompt"]:
            print("prompt:", s["prompt"])
            print("bbox:", s["shapes"][0]["bbox"])
            print("stroke数量:", len(s["strokes"]))
            print("前5笔:")
            for step in s["strokes"][:5]:
                print(f"  {step['pen_state']:8s}  dx={step['dx']:.4f}  dy={step['dy']:.4f}")
            print("...")
            print("最后3笔:")
            for step in s["strokes"][-3:]:
                print(f"  {step['pen_state']:8s}  dx={step['dx']:.4f}  dy={step['dy']:.4f}")
            break
