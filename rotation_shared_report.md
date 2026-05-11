# 旋转 + 共享参数实验报告

## 背景
我们想验证一个很具体的问题：64 层纯二维旋转矩阵堆叠起来，参数共享时是否还能学会几何变换。这类结构本质上很像 RNN 的递推，只不过“时间步”换成了“层数”。

## 任务
给定输入点 `h0 = [x, y]` 和随机旋转角 `theta`，预测旋转后的点。后续又扩展到“旋转 + 缩放”任务，目标变成：

```text
target = scale * R(theta) @ [x, y]
```

其中 `theta ∈ [0, 2π]`，`scale > 0`。

## 实验架构
模型采用 64 层堆叠结构，每层都只允许使用这种旋转矩阵：

```text
[ cosθ  -sinθ
  sinθ   cosθ ]
```

每层形式为：

```text
h_i = act( s_i * R(θ_i) h_{i-1} )
```

其中：

- `θ_i = a_i * theta + b_i`
- `s_i` 是可选缩放系数
- `act` 使用 `scaled_tanh`
- `shared_params=True` 时，64 层共享同一个 block，重复调用 64 次
- 损失函数使用 `MSE`

## 数据
生成了两份数据：

- 纯旋转：`generated_data/rotation_points/...jsonl`
- 旋转 + 缩放：`generated_data/rotation_scale_points/...jsonl`

相关脚本：

- [generate_rotation_points_dataset.py](dataset_code/generate_rotation_points_dataset.py)
- [generate_rotation_scale_points_dataset.py](dataset_code/generate_rotation_scale_points_dataset.py)

## 结果
### 1. 纯旋转 + 共享参数
- 64 层，`shared_params=True`
- `scaled_tanh(scale=1000)`
- `limit=20000`
- 最终验证集：
  - `MSE = 6.72e-4`
  - `MAE = 0.01985`

### 2. 旋转 + 缩放 + 共享参数
- 64 层，`shared_params=True`
- `use_scale=True`
- `scaled_tanh(scale=1000)`
- `limit=20000`
- 最终验证集：
  - `MSE = 3.17e-7`
  - `MAE = 4.46e-4`

## 观察
- 激活太强时，比如 `scaled_tanh(scale=10)`，64 层会明显难训，几何幅度容易被压坏。
- 把激活调得接近恒等，比如 `scale=1000`，模型就能稳定学会。
- 共享参数版本没有把任务搞崩，反而能很好拟合，说明“64 次重复同一个旋转 block”是可行的。
- 加上缩放分支后，模型表达力更强，拟合更稳。

## 简单结论
- 这类旋转堆叠结构是能学的。
- 共享参数不是问题，激活函数对几何幅度的破坏才是关键。
- 如果只保留纯旋转矩阵，最好让非线性尽量温和。
- 加上缩放系数后，模型表达力更强，拟合更稳。

## 代码
- [rotation_stack_experiment.py](experiments/rotation_stack_experiment.py)

