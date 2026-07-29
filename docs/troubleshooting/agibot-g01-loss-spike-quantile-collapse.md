# AgiBot G01 π0.5 loss 尖峰与夹爪 quantile 塌缩排查记录

## 文档用途

本文记录 `task_7792` 在 π0.5 全参数微调早期出现巨大 loss 尖峰的现象、已经执行的排查、代码依据、修复实现和下一步处理顺序。它用于后续对话迁移和故障复现。G01 专用代码修复已经写入工作区，但尚未在训练服务器重新生成统计或完成 GPU smoke test。

后续对话可以直接引用本文，并要求：

> 请先阅读 `docs/troubleshooting/agibot-g01-loss-spike-quantile-collapse.md`，从“下一步接续清单”继续，不要重新猜测问题背景。

## 状态快照

记录日期：2026-07-30，时区 Asia/Shanghai。

记录时的代码状态：

```text
分支: codex/agibot-g01-pi05
基线提交: d2ac791 Update G01 training to 32-step horizon
当前工作区: 已加入 G01 夹爪物理范围修复，尚未提交
```

训练数据与环境：

```text
项目目录: /mnt/data/workspace/lwc/openpi
数据目录: /mnt/data/workspace/lwc/openpi/dataset/task_7792
逻辑 asset id: agibot/task_7792
HF_HOME: /mnt/data/cache/huggingface
OPENPI_DATA_HOME: /mnt/data/cache/openpi
```

记录时的 `pi05_agibot_g01` 默认训练配置：

```text
model: π0.5 JAX 全参数微调
action_dim: 32（G01 有效维度为 16）
action_horizon: 32
global batch_size: 64
num_train_steps: 100000
fsdp_devices: 2
warmup_steps: 1000
peak_lr: 2.5e-5
decay_steps: 100000
decay_lr: 2.5e-6
EMA decay: 0.99
gradient clipping global norm: 1.0
log_interval: 100
save_interval: 1000
```

## 现象

训练 loss 曲线在大约 step 600 和 step 700 的日志点上升到约 `260000`，到 step 800 左右恢复到正常量级。

需要注意：[`scripts/train.py`](../../scripts/train.py) 默认每 100 步记录一次，图中的横线是相邻日志点之间的连线。因此图像不能证明 step 600 到 700 之间的每一个 batch 都具有相同 loss；一个或少量极端 batch 也可能显著抬高一个日志窗口的平均值。

该现象不是 `num_workers` 数量造成的。worker 不足可能造成 GPU 等待和吞吐下降，但不会直接把有限的训练目标放大到这种量级。

## 数据质量检查记录

使用 `scripts/agibot_g01_data_quality.py --video-check decode` 得到：

```text
episode 总数: 911
总帧数: 1324646
通过: 846
人工复核: 63
自动排除: 2
存在 >50ms 时间戳间隔的数量: 822
```

当前 `exclude_g01.txt` 仅排除：

| Episode | 原因 |
| ---: | --- |
| 475 | `max_joint_delta=0.387088` |
| 566 | `max_action_jump=0.303978` |

63 个 `review_g01.txt` episode 并不会自动排除，仍会参与归一化统计和训练。其中约 37 个只有 `source_timestamp_gap`，另外 26 个包含关节、action 或 state 突跳：

```text
1, 7, 65, 71, 103, 133, 138, 168, 173, 250, 280, 392, 421,
445, 449, 451, 452, 458, 460, 656, 658, 662, 759, 840, 894, 908
```

这些待复核 episode 仍可能产生异常 batch，但它们不是目前最强的根因证据。排除更多 episode 也不能直接解决已经确认的零 quantile range。

## 归一化统计检查记录

检查的文件：

```text
assets/pi05_agibot_g01/agibot/task_7792/norm_stats.json
```

执行过的检查代码：

```python
import json
import numpy as np

path = "assets/pi05_agibot_g01/agibot/task_7792/norm_stats.json"

with open(path) as f:
    stats = json.load(f)["norm_stats"]

for name in ("state", "actions"):
    q01 = np.asarray(stats[name]["q01"], dtype=np.float64)
    q99 = np.asarray(stats[name]["q99"], dtype=np.float64)
    scale = q99 - q01

    print(f"\n{name}:")
    for i, (low, high, width) in enumerate(zip(q01, q99, scale)):
        print(f"dim={i:02d} q01={low:.8f} q99={high:.8f} range={width:.8f}")
```

完整结果：

```text
state:
dim=00 q01=-1.08976636 q99=0.23920528 range=1.32897164
dim=01 q01=0.46626748 q99=1.35138737 range=0.88511989
dim=02 q01=-0.02695789 q99=0.97772820 range=1.00468609
dim=03 q01=-1.35753574 q99=-0.15858149 range=1.19895426
dim=04 q01=0.38782310 q99=1.34058119 range=0.95275809
dim=05 q01=-0.00040168 q99=1.18978793 range=1.19018962
dim=06 q01=-0.56516065 q99=0.32365312 range=0.88881376
dim=07 q01=0.23205758 q99=1.48086258 range=1.24880499
dim=08 q01=-1.34709326 q99=-0.31615209 range=1.03094117
dim=09 q01=-1.91891301 q99=-0.13884372 range=1.78006929
dim=10 q01=0.43227226 q99=1.36626775 range=0.93399549
dim=11 q01=-2.62470199 q99=-0.09789525 range=2.52680675
dim=12 q01=-1.49485421 q99=0.89533974 range=2.39019395
dim=13 q01=-0.30982752 q99=2.33150425 range=2.64133177
dim=14 q01=119.99788000 q99=119.99788000 range=0.00000000
dim=15 q01=0.21600000 q99=119.97600000 range=119.76000000

actions:
dim=00 q01=-0.30991697 q99=0.11116898 range=0.42108595
dim=01 q01=-0.30433030 q99=0.25535040 range=0.55968070
dim=02 q01=-0.32865868 q99=0.18862493 range=0.51728361
dim=03 q01=-0.48041053 q99=0.14070024 range=0.62111077
dim=04 q01=-0.19965534 q99=0.20592536 range=0.40558071
dim=05 q01=-0.28706948 q99=0.44070485 range=0.72777433
dim=06 q01=-0.10699523 q99=0.21701944 range=0.32401467
dim=07 q01=-0.29499713 q99=0.35653086 range=0.65152798
dim=08 q01=-0.30516741 q99=0.34561483 range=0.65078224
dim=09 q01=-0.22377330 q99=0.25982024 range=0.48359355
dim=10 q01=-0.30057563 q99=0.29418492 range=0.59476055
dim=11 q01=-0.37887631 q99=0.44518427 range=0.82406058
dim=12 q01=-0.46385376 q99=0.62649993 range=1.09035368
dim=13 q01=-0.33839496 q99=0.31460563 range=0.65300059
dim=14 q01=0.99993400 q99=0.99993400 range=0.00000000
dim=15 q01=0.00000000 q99=0.99980000 range=0.99980000
```

## 维度含义

G01 策略维度顺序为：

```text
0:7    左臂 7 关节
7:14   右臂 7 关节
14     左夹爪
15     右夹爪
```

数据选择逻辑位于 [`src/openpi/policies/agibot_g01_policy.py`](../../src/openpi/policies/agibot_g01_policy.py)：

```text
state  = raw_state[28:42] + raw_state[0:2]
action = raw_action[16:30] + raw_action[0:2]
```

因此上面两个零 range 分别对应左夹爪 state 和左夹爪 action。

## 代码链路与放大机制

### 1. 快速统计

[`scripts/compute_agibot_g01_norm_stats_fast.py`](../../scripts/compute_agibot_g01_norm_stats_fast.py) 读取 Parquet，构造 32 帧 action chunk，将前 14 维关节转成 delta，后 2 维夹爪保持绝对值，再通过 [`openpi.shared.normalize.RunningStats`](../../src/openpi/shared/normalize.py) 计算 `q01` 和 `q99`。

当前统计实现不会检测或拒绝 `q99 - q01 == 0`。

### 2. π0.5 quantile normalization

`pi05_agibot_g01` 因模型类型是 π0.5 而使用 quantile normalization。公式位于 [`src/openpi/transforms.py`](../../src/openpi/transforms.py)：

```python
(x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
```

当前实现不会将结果裁剪到 `[-1, 1]`。

当左夹爪 action 的 `q01 == q99 == 0.999934` 时，分母实际只有 `1e-6`。任何偏离该值的少数样本都会被放大。例如：

```text
原始偏差 0.01
归一化偏差量级约为 0.01 / 1e-6 × 2 = 20000
```

### 3. flow matching loss

[`src/openpi/models/pi0.py`](../../src/openpi/models/pi0.py) 使用：

```python
u_t = noise - actions
loss = mean(square(v_t - u_t))
```

极大的归一化 action 会直接进入 flow matching 目标 `u_t`，平方后产生巨大 loss。

[`scripts/train.py`](../../scripts/train.py) 对 batch、horizon 和 action 维度继续求平均，并记录：

```text
loss
grad_norm
param_norm
```

[`src/openpi/training/optimizer.py`](../../src/openpi/training/optimizer.py) 使用 `clip_by_global_norm(1.0)`。这可能解释模型在尖峰后没有立即数值发散，但梯度裁剪不能修复错误的目标尺度。

## 当前结论

### 已确认事实

1. `state dim 14` 的 `q01` 与 `q99` 完全相等。
2. `actions dim 14` 的 `q01` 与 `q99` 完全相等。
3. quantile normalization 使用 `1e-6` 作为唯一分母保护，不做 range 校验，也不裁剪输出。
4. 左夹爪少数偏离主值的样本可以被放大很多个数量级。
5. 当前只有 episode 475 和 566 被自动排除，63 个待复核 episode 仍参与训练。
6. 用户确认所有 G01 数据集的左右夹爪 state 物理范围均为 `0～120`。
7. 用户确认所有 G01 数据集的左右夹爪 action 取值均为 `0` 或 `1`。

### 高置信度判断

`actions dim 14` 的 quantile range 塌缩是 step 600～700 loss 尖峰的主要原因。该判断有明确的数值和代码链路支持，但尚未把尖峰 batch 反查到具体 episode。

### 尚未验证

1. 左夹爪各取值的实际频率，尤其是偏离 `0.999934` 的样本占比。
2. step 600、700 对应的 `grad_norm` 和 `param_norm`。
3. 尖峰 batch 是否来自 26 个运动异常待复核 episode。
4. 修复后生成的 `norm_stats.json` 尚未在训练服务器检查。
5. 修复后的真实数据训练尚未执行 GPU smoke test。

## 原始夹爪范围扫描代码

以下扫描不再用于决定物理范围；物理范围已经由用户确认。它仍可作为数据分布审计工具，在需要统计左右夹爪取值比例时执行：

```bash
source /mnt/data/workspace/lwc/openpi/env_openpi.sh

python - <<'PY'
from pathlib import Path

import numpy as np
import polars as pl

root = Path("/mnt/data/workspace/lwc/openpi/dataset/task_7792")
values = {
    "state_left": [],
    "state_right": [],
    "action_left": [],
    "action_right": [],
}

for path in sorted((root / "data").glob("chunk-*/*.parquet")):
    frame = pl.read_parquet(
        path,
        columns=["observation.state", "action"],
    )
    state = np.stack(frame["observation.state"].to_list())
    action = np.stack(frame["action"].to_list())

    values["state_left"].append(state[:, 0])
    values["state_right"].append(state[:, 1])
    values["action_left"].append(action[:, 0])
    values["action_right"].append(action[:, 1])

quantiles = [0, 0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999, 1]

for name, chunks in values.items():
    value = np.concatenate(chunks)
    print(f"\n{name}:")
    print("finite:", np.all(np.isfinite(value)))
    print("quantiles:", dict(zip(quantiles, np.quantile(value, quantiles))))
    print("fraction <= 0.5:", np.mean(value <= 0.5))
    print("fraction >= 0.99:", np.mean(value >= 0.99))
PY
```

如执行，应保存输出，用于解释左夹爪类别不均衡程度。

## 已实施修复

用户已经确认所有 G01 数据集使用：

```text
state 左右夹爪范围: 0～120
action 左右夹爪范围: 0～1
```

当前工作区已经增加 G01 专用后处理，覆盖：

```python
state.q01[14:16] = 0
state.q99[14:16] = 120
actions.q01[14:16] = 0
actions.q99[14:16] = 1
```

预期映射：

```text
state:  0 -> -1, 120 -> 1
action: 0 -> -1,   1 -> 1
```

修复集中在 `src/openpi/policies/agibot_g01_policy.py` 的 `apply_gripper_physical_norm_ranges`，并接入：

```text
scripts/compute_agibot_g01_norm_stats_fast.py  单任务快速统计
scripts/agibot_g01_multi_train.py               多任务统计
scripts/compute_norm_stats.py                    通用统计
src/openpi/training/config.py                    训练/推理加载旧统计时的运行时保护
```

该逻辑只由 `LeRobotAgiBotG01DataConfig` 和 G01 统计入口调用，没有修改其他机器人共用的全局 `Normalize` 公式。

### 不建议只做全局裁剪

简单把 quantile 输出裁剪到 `[-1,1]` 可以防止数值爆炸，但当 `q01 == q99` 时，主值和稀有左夹爪值可能被压到同一端，丢失开合监督信号。因此裁剪只能作为额外安全保护，不能替代退化维度的正确范围。

## 修复后必须增加的验证

1. 统计脚本检测所有 `q99-q01`，发现零 range 时明确报错或执行有记录的机器人专用回退。
2. 左右夹爪 state/action 的归一化结果必须 finite。
3. 已确认物理端点应映射到预期的 `-1/1`。
4. `Normalize` 与 `Unnormalize` 对 G01 夹爪执行 round-trip。
5. 真实 episode 构造出的 `(32, 16)` action chunk，其归一化绝对值必须在可解释范围内。
6. 用真实数据跑短训练，确认 `loss`、`grad_norm`、`param_norm` finite 且无同类尖峰。

## 修复后的统计与训练流程

修复统计代码后，必须重新计算 32 帧统计：

```bash
uv run scripts/compute_agibot_g01_norm_stats_fast.py \
  --dataset-root /mnt/data/workspace/lwc/openpi/dataset/task_7792 \
  --asset-id agibot/task_7792 \
  --action-horizon 32 \
  --exclude-file reports/g01_preflight/task_7792/exclude_g01.txt
```

重新执行本文的 quantile 检查代码，确认：

```text
state dim 14/15 range > 0
actions dim 14/15 range > 0
所有 q01、q99 和 range 均 finite
```

然后使用新的实验名从 π0.5 base 重新训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_agibot_g01 \
  --data.repo-id agibot/task_7792 \
  --data.dataset-root /mnt/data/workspace/lwc/openpi/dataset/task_7792 \
  --data.exclude-file reports/g01_preflight/task_7792/exclude_g01.txt \
  --exp-name g01_task_7792_h32_b64_100k_normfix \
  --batch-size 64 \
  --fsdp-devices 2 \
  --num-train-steps 100000 \
  --overwrite
```

修改归一化统计后不能从当前异常训练 checkpoint 使用 `--resume`，因为训练目标尺度已经改变。

## 下一步接续清单

后续对话从这里继续：

1. 提交并部署当前 G01 夹爪物理范围修复。
2. 在训练服务器重新生成 `task_7792` 的 32 帧统计。
3. 验证 state dim 14/15 range 为 120，actions dim 14/15 range 为 1。
4. 使用新实验名从 base checkpoint 启动短 smoke test，不恢复旧训练。
5. 检查 `loss`、`grad_norm` 和 `param_norm` 后再启动 100000 步正式训练。
6. 如尖峰仍存在，再反查 26 个运动异常待复核 episode。

当前最重要的边界是：夹爪物理范围已经确认，代码修复已经实施；服务器统计重算和 GPU 训练验证仍未完成。
