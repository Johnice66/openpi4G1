# AgiBot G01 π0.5 适配

## 状态

本文档描述当前工作区中尚未提交的 G01 实现。其变换和控制工具代码已通过语法检查及纯 NumPy 冒烟检查，但当前环境尚未运行完整 LeRobot 解码、JAX 训练、检查点服务、ROS2 连接或真机执行。

## 文件与职责

| 文件 | 职责 |
| --- | --- |
| [`src/openpi/policies/agibot_g01_policy.py`](../../src/openpi/policies/agibot_g01_policy.py) | 数据集/推理输入变换和模型输出变换 |
| [`src/openpi/training/config.py`](../../src/openpi/training/config.py) | 注册 `LeRobotAgiBotG01DataConfig` 和 `pi05_agibot_g01` |
| [`src/openpi/training/data_loader.py`](../../src/openpi/training/data_loader.py) | 显式本地 LeRobot 根目录和禁止下载保护 |
| [`scripts/compute_norm_stats.py`](../../scripts/compute_norm_stats.py) | 归一化统计命令的 `--dataset-root` 覆盖项 |
| [`scripts/compute_agibot_g01_norm_stats_fast.py`](../../scripts/compute_agibot_g01_norm_stats_fast.py) | G01 parquet 快速归一化统计，默认按本地任务目录推断 `asset_id` |
| [`scripts/serve_policy.py`](../../scripts/serve_policy.py) | checkpoint 推理服务；支持 `--policy.asset-id` 覆盖 norm stats asset |
| [`examples/agibot_g01/main.py`](../../examples/agibot_g01/main.py) | ROS2 observation 与命令节点 |
| [`examples/agibot_g01/control_utils.py`](../../examples/agibot_g01/control_utils.py) | shape 校验、重采样、EMA、融合和关节限制器 |

## 数据集与策略契约

数据集格式为 LeRobot v2.1，包含三个视频字段、163 维 `observation.state` 和 36 维 `action`。策略使用 16 维 state/action 空间：

| 策略切片 | 原始 state 切片 | 原始 action 切片 | 含义 | 表示方式 |
| --- | --- | --- | --- | --- |
| `[0:7]` | `[28:35]` | `[16:23]` | 左臂关节 | 模型训练期间为相对动作 |
| `[7:14]` | `[35:42]` | `[23:30]` | 右臂关节 | 模型训练期间为相对动作 |
| `[14:15]` | `[0:1]` | `[0:1]` | 左夹爪 | 绝对动作 |
| `[15:16]` | `[1:2]` | `[1:2]` | 右夹爪 | 绝对动作 |

`AgiBotG01Inputs` 既接受原始数据集数组，也接受推理端已组装的 16 维 state。原始 action 仅在训练期间选取。该变换会拒绝非预期维度，以及包含非有限值的 state/action。

LeRobot 视频帧是 `[0,1]` 范围内的 CHW float 数组；变换会将其转换为 HWC uint8。ROS 直接提供 HWC uint8 RGB。相机映射如下：

| 数据集/客户端名称 | 模型图像字段 |
| --- | --- |
| `top_head` | `base_0_rgb` |
| `hand_left` | `left_wrist_0_rgb` |
| `hand_right` | `right_wrist_0_rgb` |

三个 mask 均为 true。后续模型变换会将图像带 padding 地缩放到 224×224。

## π0.5 配置

`pi05_agibot_g01` 使用：

- `Pi0Config(pi05=True, action_dim=32, action_horizon=32)`；
- 未配置冻结过滤器，因此执行全参数微调；
- 从 `gs://openpi-assets/checkpoints/pi05_base/params` 加载 π0.5 base 参数；
- 从 LeRobot `task_index` 映射读取任务 prompt；
- 使用 `create_base_config` 为 π0.5 选择的分位数归一化；
- 训练 100,000 步，全局 batch size 默认为 64，默认通过 `fsdp_devices=2` 在双卡间启用参数分片；
- 默认使用 `agibot/task_5093` 作为逻辑 repository ID；训练其他任务时应通过 `--data.repo-id` 显式改成对应 `asset_id`，例如 `agibot/task_6030`。

物理策略维度是 16，但 `PadStatesAndActions` 会将 state 和 action 数组填充到模型维度 32。`AgiBotG01Outputs` 在采样后移除这些填充维度。

`DeltaActions` 在归一化之前，从手臂目标中减去 state 的前 14 维。推理期间，`AbsoluteActions` 会把反归一化后的当前关节 state 加回去，之后 `AgiBotG01Outputs` 才返回动作块。夹爪维度从不执行 delta 转换。

## 本地数据集行为

统计量计算和训练都要求显式提供根目录。推荐使用快速统计脚本，避免通用 LeRobot pipeline 解码视频：

```bash
TASK_ID=task_6030
DATASET_ROOT=/path/to/$TASK_ID
ASSET_ID=agibot/$TASK_ID

uv run scripts/compute_agibot_g01_norm_stats_fast.py \
  --dataset-root $DATASET_ROOT \
  --action-horizon 32

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_agibot_g01 \
  --data.repo-id $ASSET_ID \
  --data.dataset-root $DATASET_ROOT \
  --exp-name g01_$TASK_ID \
  --fsdp-devices 2 \
  --overwrite
```

快速统计脚本默认从 `DATASET_ROOT` 的目录名推断 asset id，例如 `/path/to/task_6030` 对应 `agibot/task_6030`。如果目录名和 asset id 不一致，使用 `--asset-id` 显式覆盖。加载器会在构造 LeRobot 对象之前检查 `<root>/meta/info.json`。由于 G01 数据配置将数据集标记为仅本地，未提供根目录时会直接失败，而不会尝试从 Hub 获取默认 repo id。

当 `num_train_steps=100000` 时，训练循环的 step index 为 `0..99999`，因此最后强制保存的检查点是 `99999`。默认 `keep_period=5000` 时，保留的周期检查点目录可能包括 `95000` 等步骤。

## 服务与 observation 协议

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_agibot_g01 \
  --policy.dir checkpoints/pi05_agibot_g01/g01_task_6030/99999 \
  --policy.asset-id agibot/task_6030 \
  --port 8000
```

ROS2 客户端发送：

```text
images.top_head    H×W×3 uint8 RGB
images.hand_left   H×W×3 uint8 RGB
images.hand_right  H×W×3 uint8 RGB
state              float32[16]，顺序为左臂关节、右臂关节、左夹爪、右夹爪
prompt             完整任务指令字符串
```

除非显式禁用元数据检查，否则客户端要求服务端元数据满足 `robot_type=agibot_g01`、`dataset_fps=30`、`action_horizon=32` 和 `policy_action_dim=16`。

## ROS2 接口

| 方向 | Topic | 消息 | 用途 |
| --- | --- | --- | --- |
| 输入 | `/camera/head_color` | `sensor_msgs/Image` | 头部 RGB |
| 输入 | `/camera/hand_left_color` | `sensor_msgs/Image` | 左手 RGB |
| 输入 | `/camera/hand_right_color` | `sensor_msgs/Image` | 右手 RGB |
| 输入 | `/hal/arm_joint_state` | `sensor_msgs/JointState` | 前 14 个位置值 |
| 输入 | `/hal/left_ee_data` | `genie_msgs/EndState` | 左夹爪位置 |
| 输入 | `/hal/right_ee_data` | `genie_msgs/EndState` | 右夹爪位置 |
| 输出 | `/wbc/arm_command` | `sensor_msgs/JointState` | 14 个绝对关节位置 |
| 输出 | `/wbc/left_ee_command` | `sensor_msgs/JointState` | 左夹爪命令 |
| 输出 | `/wbc/right_ee_command` | `sensor_msgs/JointState` | 右夹爪命令 |

## 控制与安全流程

只有指定 `--enable-control` 时才会创建命令 publisher；默认仅观察。启用确认时，操作员必须在 ROS 初始化前输入 `ENABLE`。

对于每个动作块，客户端会：

1. 校验 shape 为 `(T,16)`，拒绝空输出或非有限值；
2. 默认只保留模型输出的前八步；
3. 从 30 Hz 线性重采样到 60 Hz；
4. 在 14 个控制器 step 内，从测得的起始位姿平滑融合到手臂目标；
5. 应用 EMA，以及 `JointSafetyLimiter` 的步长、速度和加速度限制；
6. 分别缩放并裁剪左右夹爪命令；
7. 发布五次最终保持命令，并在跟踪误差超过 0.03 rad 时报告。

默认夹爪输出限制遵循已确认的数据集动作范围：左侧 `[0,0.0018111112]`，右侧 `[0,1]`。两侧仍可独立配置。客户端设置 20 秒接收超时；异常会停止主循环并进入 ROS shutdown，不会继续执行下一动作块。

这些控制属于软件保护，不提供碰撞规避、工作空间约束或急停能力。

## 验证状态

自动化测试覆盖原始数据切片、图像布局、关节 delta/absolute 往返转换、padding 移除、元数据检查、动作 shape、NaN 拒绝、重采样、限制器行为和本地根目录禁止下载行为。设置 `OPENPI_G01_TEST_DATASET_ROOT` 后可以运行真实数据集手动测试，但当前环境尚未执行。
