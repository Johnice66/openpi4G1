# AgiBot G01 π0.5 微调与部署

本适配直接使用 Genie Studio `a2d` LeRobot v2.1 数据集，无需重写数据。策略控制左臂、右臂、左夹爪和右夹爪；头部、腰部和底盘字段不属于策略动作空间。

## 训练

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

## 策略服务器

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_agibot_g01 \
  --policy.dir checkpoints/pi05_agibot_g01/g01_task_5093/29999 \
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
