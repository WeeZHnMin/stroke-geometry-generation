# HeteroKV-2DRoPE-Distance Attention

## 1. 方案名称

当前这套建模方案命名为：

**HeteroKV-2DRoPE-Distance Attention**

这个名字对应了模型中的四个核心设计：

1. **HeteroKV**
   - Key / Value 不是单一路径，而是由静态分支与趋势分支共同构成。
   - 趋势分支通过因果卷积提取局部动作走势。

2. **2DRoPE**
   - 在 self-attention 中，使用二维绝对坐标 `coords=(x, y)` 对 `Q/K` 施加二维旋转位置编码。

3. **Distance Attention**
   - 在 attention score 上额外叠加由几何距离产生的偏置项 `distance bias`。

4. **Action Triple Modeling**
   - 输出不是单一大词表 token，而是三个分类头：
     - `dx_head`
     - `dy_head`
     - `pen_head`


## 2. 任务定义

当前任务把一条绘图动作序列建模为离散三元组序列：

- `dx`
- `dy`
- `pen_state`

每个真实动作 step 会被展开为 3 个自回归位置：

1. 第 1 个位置预测 `dx`
2. 第 2 个位置预测 `dy`
3. 第 3 个位置预测 `pen_state`

因此，模型处理的不是“一步一个大 token”，而是：

`[dx_1, dy_1, pen_1, dx_2, dy_2, pen_2, ...]`

其中：

- `dx` 和 `dy` 都通过固定 bin 离散化
- `pen_state` 是有限类别标签


## 3. 输入与输出

### 3.1 输入

训练时，输入主要包括：

1. `decoder_input_ids`
   - 右移后的动作 token 序列
   - 第一个位置是 `start_id`
   - 后面是上一个时刻的真实 token

2. `coords`
   - 每个自回归位置对应的绝对二维坐标 `(x, y)`
   - 这不是 token，而是浮点坐标
   - 用于：
     - 2D RoPE
     - distance bias

3. `prompt`
   - 下游文本条件训练时使用
   - 纯解码器预训练时可关闭


### 3.2 输出

模型最终输出三个分类头：

1. `dx_logits`
2. `dy_logits`
3. `pen_logits`

其中：

- `dx_logits` 只用于 `dx` 位置
- `dy_logits` 只用于 `dy` 位置
- `pen_logits` 只用于 `pen` 位置


## 4. 前向传播

下面按完整文本条件版本说明前向传播。纯解码器预训练只是去掉文本侧与 cross-attention。

### 4.1 文本编码

如果启用文本条件：

1. `prompt` 输入冻结的中文文本编码器 `FrozenChineseTextEncoder`
2. 输出文本上下文 `context`
3. 若文本编码器隐层维度与 decoder 的 `d_model` 不一致，则经过 `context_proj`

这部分由：

- `TextConditionedActionModel.encode_text`
- `stroke_baseline/action_model.py`

负责。


### 4.2 动作 token 嵌入

`decoder_input_ids` 进入：

1. `token_emb`
2. `pos_emb`

然后两者相加得到 decoder 输入隐表示：

`x = token_emb(decoder_input_ids) + pos_emb(position_ids)`

这一步定义在：

- `ActionTokenDecoder.forward`
- `stroke_baseline/action_model.py`


### 4.3 Self-Attention 中的几何建模

每个 decoder block 的第一部分是 self-attention。

#### 4.3.1 Q / K / V 的构造

以 `hetero` 版本为例：

1. `Q` 来自线性投影 `q_proj(x)`
2. `K/V` 分成两条路径：
   - 静态路径：`k_proj_static / v_proj_static`
   - 趋势路径：`causal_conv -> SiLU -> k_proj_trend / v_proj_trend`
3. 两条路径相加形成最终 `K/V`

也就是：

- `K = K_static + scale_k * K_trend`
- `V = V_static + scale_v * V_trend`

对应实现：

- `HeterogeneousCachedTokenSelfAttention`
- `stroke_baseline/action_model.py`


#### 4.3.2 2D RoPE

如果传入了 `coords`，则对 `Q/K` 施加二维旋转位置编码。

具体做法：

1. 将每个 attention head 的维度一分为二
   - 前半部分对应 `x`
   - 后半部分对应 `y`

2. 分别用：
   - `coords[..., 0]` 生成 `x` 方向旋转角
   - `coords[..., 1]` 生成 `y` 方向旋转角

3. 用这些旋转角对 `Q/K` 做正交旋转

这样，attention 内积中自然会引入二维相对位置信息。

对应实现：

- `rotate_half`
- `apply_2d_rotary_pos_emb`
- `stroke_baseline/action_model.py`


#### 4.3.3 Distance Bias

如果开启 `use_distance_bias`：

1. 根据 `coords` 计算点与点之间的欧氏距离
2. 把距离送进一个小 MLP
3. 输出每个 head 的 bias
4. 把 bias 加到 attention score 上

公式可以写成：

`score(i, j) = (q_i · k_j + b(dist(i, j))) / sqrt(d)`

其中：

- `dist(i, j)` 是第 `i` 个位置与第 `j` 个位置的二维几何距离
- `b(.)` 是一个可学习 MLP

对应实现：

- `_distance_bias`
- `_attend`
- `HeterogeneousCachedTokenSelfAttention`
- `stroke_baseline/action_model.py`


### 4.4 Cross-Attention

如果启用文本条件训练：

1. self-attention 后的 hidden 作为 query
2. 文本 `context` 作为 key / value
3. 执行 cross-attention

如果是纯解码器预训练，则：

- `use_cross_attn = False`
- 这一步被跳过

对应实现：

- `ActionDecoderBlock`
- `stroke_baseline/action_model.py`


### 4.5 FFN

每一层 block 的最后还有前馈网络：

1. `Linear(d_model -> d_model * ff_mult)`
2. `GELU`
3. `Dropout`
4. `Linear(d_model * ff_mult -> d_model)`

然后通过残差连接回到主干。


### 4.6 三头输出

最后经过 `LayerNorm` 后，输出三个分类头：

1. `dx_head(hidden)`
2. `dy_head(hidden)`
3. `pen_head(hidden)`

对应实现：

- `ActionTokenDecoder`
- `stroke_baseline/action_model.py`


## 5. 梯度下降与损失函数

### 5.1 三头交叉熵

训练损失是三头分类损失：

1. 在 `phase % 3 == 0` 的位置，用 `dx_logits` 监督 `dx_target`
2. 在 `phase % 3 == 1` 的位置，用 `dy_logits` 监督 `dy_target`
3. 在 `phase % 3 == 2` 的位置，用 `pen_logits` 监督 `pen_target`

然后将三部分 loss 取平均：

`loss = mean(dx_loss, dy_loss, pen_loss)`

实现位置：

- `compute_loss`
- `stroke_baseline/train_action_tokens.py`


### 5.2 Two-Stage 情况

如果使用 two-stage 模式，还会额外有：

- `start_loss = MSE(start_pred, start_position)`

总损失会变成：

`loss_total = mean(dx_loss, dy_loss, pen_loss) + start_loss`


### 5.3 梯度传播

每个训练 batch 的标准流程是：

1. 前向传播得到 `dx_logits / dy_logits / pen_logits`
2. 计算 loss
3. `loss.backward()`
4. 对可训练参数做梯度裁剪
5. `optimizer.step()`

优化器使用：

- `AdamW`

代码位置：

- `train_one_epoch`
- `stroke_baseline/train_action_tokens.py`


### 5.4 哪些参数会更新

#### 文本条件下游训练

默认情况下：

- 文本编码器参与前向
- 但 `FrozenChineseTextEncoder` 不更新

会更新的主要包括：

- decoder 参数
- `context_proj`
- `start_head`


#### 纯解码器预训练

当启用：

- `--decoder-only-pretrain`

时：

- 不使用文本编码器
- 不使用 cross-attention
- 只训练 decoder 主干与相关输出头


## 6. 训练模式

### 6.1 纯解码器预训练

用途：

- 学动作序列自身的自回归先验

特征：

- `use_cross_attn = False`
- `prompts = None`
- 输入只依赖动作 token 和 `coords`


### 6.2 文本条件下游微调

用途：

- 从文本生成动作序列

特征：

- `use_cross_attn = True`
- 使用文本编码器输出 `context`
- decoder 通过 cross-attention 读取文本条件


## 7. 涉及的核心代码文件

### 7.1 模型定义

- [action_model.py](/abs/path/c:/Users/34619/Desktop/Program/stroke-geometry-generation/stroke_baseline/action_model.py)

主要包含：

- `ActionDecoderConfig`
- `apply_2d_rotary_pos_emb`
- `HeterogeneousCachedTokenSelfAttention`
- `ActionDecoderBlock`
- `ActionTokenDecoder`
- `TextConditionedActionModel`


### 7.2 动作数据集

- [action_dataset.py](/abs/path/c:/Users/34619/Desktop/Program/stroke-geometry-generation/stroke_baseline/action_dataset.py)

主要包含：

- `ActionTokenJsonlDataset`
- `TwoStageActionTokenJsonlDataset`
- `coords` 构造
- jsonl 索引式懒加载


### 7.3 训练脚本

- [train_action_tokens.py](/abs/path/c:/Users/34619/Desktop/Program/stroke-geometry-generation/stroke_baseline/train_action_tokens.py)

主要包含：

- `compute_loss`
- `train_one_epoch`
- `evaluate`
- checkpoint 保存
- 纯解码器 / 文本条件两种训练模式


### 7.4 动作 tokenizer

- [action_tokenizer.py](/abs/path/c:/Users/34619/Desktop/Program/stroke-geometry-generation/stroke_baseline/action_tokenizer.py)

主要负责：

- `dx/dy/pen` 的固定离散化
- token id 编码与解码


### 7.5 采样与评估

- [sample_action_tokens.py](/abs/path/c:/Users/34619/Desktop/Program/stroke-geometry-generation/stroke_baseline/sample_action_tokens.py)
- [eval_action_prefix_rollout.py](/abs/path/c:/Users/34619/Desktop/Program/stroke-geometry-generation/stroke_baseline/eval_action_prefix_rollout.py)

主要负责：

- checkpoint 加载
- 自回归 rollout
- prefix replay 评估


### 7.6 数据导出脚本

- [export_hetero_kv_distance_dataset.py](/abs/path/c:/Users/34619/Desktop/Program/stroke-geometry-generation/dataset_code/export_hetero_kv_distance_dataset.py)

主要负责：

- 将原始 stroke 数据导出为：
  - `decoder_input_ids`
  - `target_ids`
  - `target_mask`
  - `coords`


## 8. 当前方案的优点与风险

### 优点

1. 把 `dx / dy / pen` 拆成三头后，训练目标更稳定
2. `coords` 同时进入：
   - 2D RoPE
   - distance bias
3. `HeteroKV` 能兼顾：
   - 局部几何趋势
   - 全局静态记忆
4. 可统一支持：
   - decoder-only 预训练
   - 文本条件下游微调


### 风险

1. 如果数据中 `move` 段过长，模型会学到重复平移而不是有效绘制
2. 2D RoPE 与 distance bias 都依赖 `coords` 质量
3. 纯解码器预训练效果高度依赖动作序列本身是否健康


## 9. 一句话总结

**HeteroKV-2DRoPE-Distance Attention** 是一个面向离散 `dx/dy/pen_state` 动作序列的自回归 Transformer 解码框架，它通过 `HeteroKV + 2D RoPE + Distance Bias` 在注意力层中显式注入二维几何结构，并通过三头分类目标稳定地学习绘图动作序列。
