# Stroke Baseline

Minimal trainable baseline for text-to-stroke generation.

## Tokenizer

The default version uses HuggingFace `google-bert/bert-base-chinese`, a Chinese WordPiece tokenizer with a 21,128-token vocabulary. It supports Chinese prompts directly and is still compact compared with common LLM tokenizers.

The original `CharTokenizer` is still available for tiny sanity checks:

```powershell
python -m stroke_baseline.train --tokenizer-type char
```

A tokenizer only needs to expose:

- `pad_id`
- `vocab_size`
- `encode(text, max_len)`
- `save(path)` / `load(path)`

## Train

```powershell
python -m stroke_baseline.train --epochs 5 --batch-size 64
```

For a tiny overfit sanity check:

```powershell
python -m stroke_baseline.train --data generated_data\toy_stroke_dataset.jsonl --epochs 200 --batch-size 2 --output-dir runs\stroke_toy
```

## Sample

```powershell
python -m stroke_baseline.sample --checkpoint-dir runs\stroke_baseline --prompt "draw a medium rectangle on the right" --png runs\stroke_baseline\sample_custom.png
```
