# AgiBot G01 π0.5 微调与部署

本适配直接使用 Genie Studio `a2d` LeRobot v2.1 数据集，无需重写数据。策略控制左臂、右臂、左夹爪和右夹爪；头部、腰部和底盘字段不属于策略动作空间。

## Action chunk 默认值

`pi05_agibot_g01` 默认使用 `Pi0Config(pi05=True, action_dim=32, action_horizon=16)`。对 G01 来说，策略有效 action 是 16 维；模型内部会把 16 维 G01 action padding 到 32 维，推理输出再恢复为 16 维。

单任务和多任务训练默认都是 16 步 action chunk：

```text
模型返回 actions shape: (16, 16)
每步 action 维度: 左臂 7 + 右臂 7 + 左夹爪 1 + 右夹爪 1
ROS2 客户端默认执行: 前 8 步，也就是 --execute-horizon 8
```

不要单独修改 `--action-horizon`。如果确实要改 action chunk，需要同时修改模型配置、归一化统计、训练数据 chunk 构造和 ROS2 推理端执行逻辑。

## 单任务训练

在受支持的 Linux/NVIDIA 机器上安装 openpi，然后从本地数据集计算归一化统计量：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_agibot_g01 \
  --dataset-root /path/to/task_5093
```

启动 π0.5 全参数微调。根据可用 GPU 选择 `--batch-size` 和 `--fsdp-devices`：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_agibot_g01 \
  --data.dataset-root /path/to/task_5093 \
  --exp-name g01_task_5093 \
  --overwrite
```

策略 state/action 的配置顺序如下：

```text
左臂关节 [0:7]，右臂关节 [7:14]，左夹爪 [14]，右夹爪 [15]
```

模型学习相对关节动作和绝对夹爪动作。策略输出变换会在返回客户端前恢复绝对关节目标。

默认 `num_train_steps=30000` 时，当前训练脚本最终保存的 step 通常是 `29999`，因为训练循环执行 `0..29999`。启动策略服务器时使用实际生成的 checkpoint step 目录。

## 多任务训练

多任务训练使用 `scripts/agibot_g01_multi_train.py`，不会修改原始 LeRobot 数据集，只在训练时把多个本地数据集混合起来。每个数据集都必须是同一套 G01 数据格式：本地 root 下存在 `meta/info.json`，原始 state/action schema 一致，并且相机、state、action 字段能通过当前 G01 transform 读取。

推荐先定义路径变量：

```bash
TASK_5093=/path/to/task_5093
TASK_6030=/path/to/task_6030
TASK_7001=/path/to/task_7001

ASSET_ID=agibot/task_5093
ASSETS_BASE=./assets_g01_multitask
```

`ASSET_ID=agibot/task_5093` 是为了不改推理配置：`serve_policy.py --policy.config pi05_agibot_g01` 默认会按 `agibot/task_5093` 查找 checkpoint 内的 norm stats。用单独的 `ASSETS_BASE` 可以避免覆盖普通单任务的 `./assets`。如果你改成自定义 asset id，例如 `agibot/g01_mix_5093_6030`，就需要同步新增对应训练配置，或让推理入口显式使用这个 asset id。

先计算混合归一化统计：

```bash
uv run scripts/agibot_g01_multi_train.py compute-norm \
  --dataset agibot/task_5093=$TASK_5093 \
  --dataset agibot/task_6030=$TASK_6030 \
  --dataset agibot/task_7001=$TASK_7001 \
  --asset-id $ASSET_ID \
  --output-dir $ASSETS_BASE/pi05_agibot_g01/$ASSET_ID
```

再启动多任务 π0.5 全参数微调：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/agibot_g01_multi_train.py train \
  --dataset agibot/task_5093=$TASK_5093 \
  --dataset agibot/task_6030=$TASK_6030 \
  --dataset agibot/task_7001=$TASK_7001 \
  --asset-id $ASSET_ID \
  --assets-base-dir $ASSETS_BASE \
  --exp-name g01_multitask_5093_6030_7001 \
  --batch-size 64 \
  --num-train-steps 30000 \
  --overwrite
```

默认采样策略是按各数据集帧数比例采样。常用覆盖方式：

```bash
# 每个任务等概率采样，不管各自帧数多少。
--equal-dataset-sampling

# 按给定任务权重采样，顺序必须和 --dataset 顺序一致，脚本会自动归一化。
--sampling-weight 1.0 \
--sampling-weight 2.0 \
--sampling-weight 1.0
```

多任务 smoke test 可以先限制统计帧数并跑短训练：

```bash
uv run scripts/agibot_g01_multi_train.py compute-norm \
  --dataset agibot/task_5093=$TASK_5093 \
  --dataset agibot/task_6030=$TASK_6030 \
  --asset-id $ASSET_ID \
  --output-dir $ASSETS_BASE/pi05_agibot_g01/$ASSET_ID \
  --max-frames 5000
```

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/agibot_g01_multi_train.py train \
  --dataset agibot/task_5093=$TASK_5093 \
  --dataset agibot/task_6030=$TASK_6030 \
  --asset-id $ASSET_ID \
  --assets-base-dir $ASSETS_BASE \
  --exp-name g01_multitask_smoke \
  --batch-size 16 \
  --num-train-steps 100 \
  --wandb-enabled false \
  --overwrite
```

多 GPU 训练仍走 JAX trainer：`--batch-size` 必须能被 `jax.device_count()` 整除，`--fsdp-devices` 也要按实际 GPU 数配置。GPU 数量、显存和最终 batch size 不确定时，先用较小 batch 跑 smoke test，再放大。

## 策略服务器

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_agibot_g01 \
  --policy.dir checkpoints/pi05_agibot_g01/g01_task_5093/29999 \
  --port 8000
```

多任务 checkpoint 也使用同一个 `pi05_agibot_g01` policy config，前提是训练时按上面的推荐保持 `ASSET_ID=agibot/task_5093`：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_agibot_g01 \
  --policy.dir checkpoints/pi05_agibot_g01/g01_multitask_5093_6030_7001/29999 \
  --port 8000
```

## ROS2 客户端

机器人环境必须提供 `rclpy`、`cv_bridge`、`sensor_msgs`、`genie_msgs`、NumPy 和 `openpi-client` 包。首先运行仅观察推理：

```bash
python examples/agibot_g01/main.py \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --no-enable-control \
  --max-cycles 10
```

检查日志中的动作 shape 和数值范围。只有确认机器人侧急停、碰撞保护、关节顺序和夹爪单位后，才能启用命令发布：

```bash
python examples/agibot_g01/main.py \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --enable-control
```

启用控制时必须输入 `ENABLE`。只有在具备独立监管的部署中才应使用 `--no-confirm-control`。默认行为是执行模型输出的前八步，从 30 Hz 重采样到 60 Hz，在动作块边界进行融合，并施加关节步长、速度和加速度限制。
