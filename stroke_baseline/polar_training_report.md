# Polar 训练说明

本文记录的是这次 `stroke_polar_unique_15000_e16` 训练，对应的代码、数据和结果都严格对齐如下。

## 1. 这次训练对应什么

- 训练入口: `stroke_baseline/train_polar_tokens.py`
- 模型实现: `stroke_baseline/polar_model.py`, `stroke_baseline/action_model.py`, `stroke_baseline/pretrained_encoder_decoder.py`
- 数据集实现: `stroke_baseline/polar_dataset.py`
- token 定义: `stroke_baseline/polar_tokenizer.py`
- 训练结果目录: `runs/stroke_polar_unique_15000_e16`

## 2. 数据是怎么来的

这次训练用的数据不是手工标注，而是先生成原始 stroke，再转成 polar token 数据。

- 原始样本生成总入口: `dataset_code/run_polar_token_pipeline.py`
- 原始场景采样: `dataset_code/generate_quantized_grid_dataset.py` 里的 `sample_scene`
- stroke -> polar token 转换: `dataset_code/generate_polar_token_dataset_from_strokes.py`
- vocab 导出: `dataset_code/export_polar_vocab.py`
- 数据抽样可视化: `dataset_code/sample_polar_token_dataset.py`

对应的数据文件是:

- 训练数据: `generated_data/polar_tokens/polar_tokens_unique_15000.jsonl`
- vocab 文件: `generated_data/polar_tokens/polar_tokens_unique_15000_vocab.json`

这套数据里，每个样本都包含:

- `prompt`
- `strokes`
- `action_tokens`
- `decoded_polar_strokes`
- `metadata`

如果按代码链路拆开看，数据经过了三层处理:

- `dataset_code/generate_quantized_grid_dataset.py`
  负责生成连续的几何 stroke 样本。
- `dataset_code/generate_polar_token_dataset_from_strokes.py`
  负责把每个 `(dx, dy, pen_state)` 转成 polar action token，并回写误差统计。
- `stroke_baseline/polar_dataset.py`
  负责把 JSONL 样本整理成训练时真正使用的张量。

## 3. token 化方法

对应代码: `stroke_baseline/polar_tokenizer.py` 和 `dataset_code/generate_polar_token_dataset_from_strokes.py`

核心思路是把每一步 stroke action 变成一个离散 token:

- `distance_id`
- `theta_id`
- `pen_state`

然后编码成一个单 token:

`token = ((distance_id * theta_bins) + theta_id) * num_pen_states + pen_id`

这里用的是:

- 距离分桶: `0.1, 0.2, 0.3, 0.4, 0.5`
- 角度分桶: `theta_bins = 360`
- 笔状态: `move / draw / end_all`

在 `dataset_code/generate_polar_token_dataset_from_strokes.py` 里，还会把量化后的 token 再 decode 回 stroke，
并把下面这些误差写进 `metadata`:

- `polar_dx_mae`
- `polar_dy_mae`
- `polar_distance_mae`
- `polar_theta_mae_deg`

这次训练还用了 `CompactPolarTokenMapper`:

- 代码: `stroke_baseline/polar_tokenizer.py`
- 作用: 把观测到的 raw token 压成连续小词表
- 对应 vocab 文件: `generated_data/polar_tokens/polar_tokens_unique_15000_vocab.json`

这意味着训练时不是直接在“理论上的全量 polar token 空间”里学习，
而是在“这 15000 条训练样本里实际出现过的 token 集合”上学习，
这样可以减小 softmax 词表规模。

## 4. 模型与张量

### 4.1 训练张量是怎么构造的

对应代码: `stroke_baseline/polar_dataset.py`

`PolarActionJsonlDataset` 会把每个样本整理成:

- `prompt`
- `decoder_input_ids`
- `target_ids`
- `target_mask`
- `length`

具体做法是标准的自回归 shift:

- 第 0 位输入放 `start_token`
- 后面的输入放前一个 token
- `target_ids` 放当前真实 token
- 超出真实长度的位置用 `pad_token` 和 `-100`
- `target_mask` 标出有效位置

所以这个任务本质上是 next-token prediction，只不过 token 语义不是自然语言词，
而是极坐标 stroke action。

### 4.2 模型架构

对应代码:

- 外层封装: `stroke_baseline/polar_model.py`
- decoder 细节: `stroke_baseline/action_model.py`
- 冻结文本编码器: `stroke_baseline/pretrained_encoder_decoder.py`

结构是一个 text-conditioned autoregressive decoder:

- 文本端用本地 `bert-base-chinese`
- 文本编码器冻结，不参与训练
- 如果文本 hidden size 和 decoder `d_model` 不一致，就用 `context_proj` 做线性投影
- decoder 是多层 Transformer decoder
- 输入是 `token_emb + learned positional embedding`
- 自回归 self-attention 用 causal mask
- 再通过 cross-attention 读取文本上下文

更具体地说:

- `stroke_baseline/polar_model.py`
  只是一层包装，把冻结文本编码器和 action decoder 接起来。
- `stroke_baseline/action_model.py`
  真正实现了 `ActionTokenDecoder`、`ActionDecoderBlock`、`CachedTokenSelfAttention`。
- `stroke_baseline/pretrained_encoder_decoder.py`
  提供了 `FrozenChineseTextEncoder` 和 cross-attention 逻辑。

这里的位置编码不是 RoPE，也不是固定正弦编码，而是可学习的绝对位置嵌入:

- `stroke_baseline/action_model.py` 里的 `self.pos_emb = nn.Embedding(...)`
- `stroke_baseline/model.py` 里同样也是 `nn.Embedding(...)` 的绝对位置写法

### 4.3 这次模型不是直接回归 `dx, dy`

这是这套 baseline 最关键的建模选择。

不是让模型直接输出连续值:

- `dx`
- `dy`
- `pen_state`

而是先把动作离散化成 polar token，再做 token generation。

这样做的好处是:

- 训练目标统一成离散分类，loss 更直接
- 推理阶段天然支持自回归生成
- `pen_state` 不再需要单独建头
- 距离和方向被绑定进同一个 token，动作语义更完整

代价是:

- 量化误差一定存在
- token 空间设计会直接影响上限
- 数据分布不均时，长尾 token 仍然难学

## 5. 建模方法和训练思路

对应代码: `stroke_baseline/train_polar_tokens.py`

这次做的是“文本 -> 离散 polar action 序列”的监督学习：

- 输入: `prompt`
- 输出: 下一步 `action_token`
- 训练方式: teacher forcing
- 损失: `cross_entropy`
- 忽略位: `ignore_index = -100`
- 评估指标: `token_acc`

训练循环的关键实现也都在 `stroke_baseline/train_polar_tokens.py`:

- `compute_loss`
  计算 token 级交叉熵和 token accuracy。
- `train_one_epoch`
  做前向、反向、梯度裁剪和 step 级日志记录。
- `evaluate`
  在验证集上做整轮评估。
- `save_checkpoint`
  保存 decoder、context projection、tokenizer 配置和训练参数。

训练时的关键超参来自 `runs/stroke_polar_unique_15000_e16/train_args.json`:

- `epochs = 16`
- `batch_size = 16`
- `lr = 1e-4`
- `weight_decay = 0.01`
- `grad_clip = 1.0`
- `d_model = 384`
- `n_heads = 8`
- `decoder_layers = 3`
- `dropout = 0.1`
- `max_text_len = 64`
- `max_action_len = 192`
- `val_ratio = 0.1`

训练实现上的几个准确细节:

- 数据集通过 `random_split(..., generator=torch.Generator().manual_seed(args.seed))` 切成 train/val
- 优化器是 `AdamW`
- 只更新 `requires_grad=True` 的参数，所以冻结文本编码器不会被更新
- 梯度裁剪阈值是 `1.0`
- best checkpoint 的判据是 `val loss` 最小

## 6. 训练结果

对应文件:

- `runs/stroke_polar_unique_15000_e16/epoch_metrics.jsonl`
- `runs/stroke_polar_unique_15000_e16/step_metrics.jsonl`
- `runs/stroke_polar_unique_15000_e16/metrics.png`
- `runs/stroke_polar_unique_15000_e16/checkpoint.pt`

这些文件分别表示:

- `train_args.json`
  训练配置快照。
- `step_metrics.jsonl`
  按训练 step 记录的中间 loss 和 acc。
- `epoch_metrics.jsonl`
  每个 epoch 的 train/val 汇总指标。
- `metrics.png`
  根据日志画出的训练曲线。
- `checkpoint.pt`
  best 验证结果对应的模型权重。

结果摘要:

- 总训练轮数: 16
- 最佳验证轮次: epoch 14
- 最佳验证 loss: `0.9970795264903535`
- 最佳验证 acc: `0.670012393530379`
- 对应训练 loss: `0.8784857967171059`
- 对应训练 acc: `0.6989952060283643`

最后一轮:

- `train loss = 0.8430424220330343`
- `train acc = 0.7096958009976346`
- `val loss = 0.9976959551902528`
- `val acc = 0.6722211780700278`

从数值上看，这个 run 的特点是:

- 训练 loss 持续下降
- 验证 loss 在 epoch 14 左右达到最优
- epoch 16 的 `val_acc` 略高于 epoch 14，但 `val_loss` 不是最优

因此 `checkpoint.pt` 保存的是 best `val_loss` 对应的模型，不是最后一轮权重。

## 7. 采样与可视化

对应代码:

- `stroke_baseline/sample_polar_tokens.py`
- `stroke_baseline/visualize.py`

采样时会:

- 从 checkpoint 载入模型
- 自回归生成 token
- 再把 token 反解成 stroke
- 最后画成 png 或导出 json

如果要检查“数据量化本身有没有问题”，更直接的脚本不是模型采样，
而是 `dataset_code/sample_polar_token_dataset.py`。
它会直接把数据集里的 `action_tokens` decode 回 stroke，再和原 `strokes` 比较误差。

## 8. 这次训练最准确的文件对应关系

- 训练主程序: `stroke_baseline/train_polar_tokens.py`
- 训练数据读取: `stroke_baseline/polar_dataset.py`
- polar token 定义: `stroke_baseline/polar_tokenizer.py`
- polar 模型封装: `stroke_baseline/polar_model.py`
- decoder 结构: `stroke_baseline/action_model.py`
- 冻结中文文本编码器与 cross-attention: `stroke_baseline/pretrained_encoder_decoder.py`
- 训练后采样: `stroke_baseline/sample_polar_tokens.py`
- 训练数据生成总入口: `dataset_code/run_polar_token_pipeline.py`
- 连续 stroke 转 polar token: `dataset_code/generate_polar_token_dataset_from_strokes.py`
- 观测词表导出: `dataset_code/export_polar_vocab.py`
- 数据反解检查: `dataset_code/sample_polar_token_dataset.py`
- 原始几何场景采样: `dataset_code/generate_quantized_grid_dataset.py`
- 本次训练数据: `generated_data/polar_tokens/polar_tokens_unique_15000.jsonl`
- 本次训练 vocab: `generated_data/polar_tokens/polar_tokens_unique_15000_vocab.json`
- 本次训练结果目录: `runs/stroke_polar_unique_15000_e16`
