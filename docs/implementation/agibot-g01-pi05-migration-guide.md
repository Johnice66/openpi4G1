# 从官方 openpi 迁移到当前 AgiBot G01 π0.5 适配

本文档面向“拿一份官方 openpi 代码，按步骤改成当前分支实现”的场景。它不替代代码 diff，而是解释每个改动的目的、文件位置、关键实现和验证方法。

当前实现基于本仓库 `codex/agibot-g01-pi05` 分支的提交 `141b745`，相对基线是 `origin/main`。提交时间为 `2026-07-07 00:08:59 +0800`。如果官方仓库后续改动了同名文件，应优先按职责合并，而不是机械套用行号。

## 目标结果

完成迁移后，应具备以下能力：

- 使用 `pi05_agibot_g01` 配置从 `gs://openpi-assets/checkpoints/pi05_base/params` 初始化 π0.5 base 并执行 JAX 全参数微调。
- 显式从本地 LeRobot v2.1 `task_5093` 目录读取数据，不修改原始数据，也不回退到 Hugging Face Hub 下载。
- 将原始三路 RGB、163 维 state、36 维 action 和整任务语言转换为模型需要的三路图像、16 维 state、16 帧 action chunk。
- 对前 14 维手臂动作使用相对动作训练，对末 2 维夹爪保持绝对动作。
- 通过 WebSocket 策略服务器加载检查点，通过 ROS2 客户端发送 observation，并默认禁用真机控制。

## 改动总览

| 顺序 | 文件 | 改动类型 | 迁移目的 |
| --- | --- | --- | --- |
| 1 | [`src/openpi/training/config.py`](../../src/openpi/training/config.py) | 扩展数据配置 | 支持本地数据根目录、G01 数据工厂和 `pi05_agibot_g01` 注册项 |
| 2 | [`src/openpi/training/data_loader.py`](../../src/openpi/training/data_loader.py) | 扩展 LeRobot loader | 把 `dataset_root` 传给 LeRobot，并在本地数据缺失时提前失败 |
| 3 | [`scripts/compute_norm_stats.py`](../../scripts/compute_norm_stats.py) | 扩展 CLI | 允许统计量命令通过 `--dataset-root` 指向本地 `task_5093` |
| 4 | [`src/openpi/policies/agibot_g01_policy.py`](../../src/openpi/policies/agibot_g01_policy.py) | 新增策略变换 | 实现 G01 state/action 切片、图像格式转换和输出去 padding |
| 5 | [`packages/openpi-client/src/openpi_client/websocket_client_policy.py`](../../packages/openpi-client/src/openpi_client/websocket_client_policy.py) | 扩展客户端超时 | 让机器人客户端在服务端无响应时可退出 |
| 6 | [`examples/agibot_g01`](../../examples/agibot_g01) | 新增 ROS2 示例 | 实现传感器采集、远程推理、动作校验和可选安全发布 |
| 7 | [`src/openpi/policies/*agibot*test.py`](../../src/openpi/policies/agibot_g01_policy_test.py)、[`src/openpi/training/data_loader_test.py`](../../src/openpi/training/data_loader_test.py) | 新增测试 | 覆盖切片、delta/absolute、shape、非有限值、本地 root 行为 |
| 8 | [`docs`](../../docs)、[`examples/agibot_g01/README.md`](../../examples/agibot_g01/README.md) | 中文文档 | 记录当前实现、训练和部署命令 |

## 步骤 1：扩展训练配置的数据根目录

修改 [`src/openpi/training/config.py`](../../src/openpi/training/config.py)。

### 1.1 引入 G01 policy transform 模块

在已有 policy imports 旁增加：

```python
import openpi.policies.agibot_g01_policy as agibot_g01_policy
```

原因：G01 的 `DataConfigFactory` 需要引用 `AgiBotG01Inputs`、`AgiBotG01Outputs` 和 `JOINT_ACTION_MASK`。

### 1.2 给 `DataConfig` 增加本地数据字段

在 `DataConfig` 中增加：

```python
dataset_root: str | None = None
local_dataset_only: bool = False
```

语义：

- `dataset_root` 是本地 LeRobot 数据集目录，例如 `/path/to/task_5093`。
- `local_dataset_only=True` 表示该配置禁止没有 root 时自动下载。

这一步是为了让训练和统计量计算都能显式读取本地 Genie Studio `a2d` 数据，而不需要改动数据集本身。

### 1.3 给 `DataConfigFactory` 增加 CLI 可覆盖字段

在 `DataConfigFactory` 中增加：

```python
dataset_root: str | None = None
```

并在 `create_base_config` 的 `dataclasses.replace(...)` 中传入：

```python
dataset_root=self.dataset_root
```

这样 `scripts/train.py` 可以通过 Tyro 接受：

```bash
--data.dataset-root /path/to/task_5093
```

如果只改 `DataConfig`，不改 `DataConfigFactory`，CLI 覆盖不会自然进入最终 `DataConfig`。

## 步骤 2：新增 G01 的 LeRobot 数据工厂

继续修改 [`src/openpi/training/config.py`](../../src/openpi/training/config.py)，在其他 `LeRobot...DataConfig` 类附近新增 `LeRobotAgiBotG01DataConfig`。

关键结构如下：

```python
@dataclasses.dataclass(frozen=True)
class LeRobotAgiBotG01DataConfig(DataConfigFactory):
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[agibot_g01_policy.AgiBotG01Inputs()],
            outputs=[agibot_g01_policy.AgiBotG01Outputs()],
        ).push(
            inputs=[_transforms.DeltaActions(agibot_g01_policy.JOINT_ACTION_MASK)],
            outputs=[_transforms.AbsoluteActions(agibot_g01_policy.JOINT_ACTION_MASK)],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
            local_dataset_only=True,
        )
```

这里有三个容易出错的点：

1. `repack_transform` 只处理训练数据集字段，把 LeRobot 原字段映射成 policy transform 接受的通用字段。
2. `AgiBotG01Inputs` 同时被训练和推理复用，所以它不能依赖 LeRobot 专有字段名。
3. `Group.push(...)` 对 inputs 是追加，对 outputs 是插到前面；因此推理输出会先执行 `AbsoluteActions`，再执行 `AgiBotG01Outputs` 去掉模型 padding。

## 步骤 3：注册 `pi05_agibot_g01`

在 [`src/openpi/training/config.py`](../../src/openpi/training/config.py) 的 `_CONFIGS` 列表中新增：

```python
TrainConfig(
    name="pi05_agibot_g01",
    model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=16),
    data=LeRobotAgiBotG01DataConfig(
        repo_id="agibot/task_5093",
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=30_000,
    policy_metadata={
        "robot_type": "agibot_g01",
        "dataset_fps": 30,
        "action_horizon": 16,
        "policy_action_dim": 16,
        "state_order": ["left_arm_joint_position", "right_arm_joint_position", "left_gripper", "right_gripper"],
        "action_order": ["left_arm_joint_position", "right_arm_joint_position", "left_gripper", "right_gripper"],
    },
)
```

设计要点：

- `pi05=True` 让模型类型走 π0.5 路径。
- `action_dim=32` 是模型维度，实际 G01 action 维度是 16，后续通过 `PadStatesAndActions` 填充。
- `action_horizon=16` 对应 16 帧 action chunk。
- 未设置 `freeze_filter`，因此是 JAX 全参数微调。
- `prompt_from_task=True` 使用 LeRobot episode 的整任务语言，而不是 `sub_tasks`。
- `policy_metadata` 供 ROS2 客户端启动时校验服务端是否加载了正确策略。

## 步骤 4：让 LeRobot loader 支持本地 root

修改 [`src/openpi/training/data_loader.py`](../../src/openpi/training/data_loader.py) 的 `create_torch_dataset`。

在创建 `LeRobotDatasetMetadata` 之前增加本地数据检查：

```python
if data_config.local_dataset_only and data_config.dataset_root is None:
    raise ValueError(f"Dataset {repo_id!r} requires an explicit local dataset root")
if data_config.dataset_root is not None:
    dataset_root = pathlib.Path(data_config.dataset_root)
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"Local LeRobot dataset root does not contain meta/info.json: {dataset_root}"
        )
```

然后把 root 传给 LeRobot：

```python
dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=data_config.dataset_root)
dataset = lerobot_dataset.LeRobotDataset(
    data_config.repo_id,
    root=data_config.dataset_root,
    delta_timestamps={
        key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
    },
)
```

这样 `pi05_agibot_g01` 在没有 `--data.dataset-root` 时会直接失败，不会尝试下载 `agibot/task_5093`。这符合“本地数据、不修改原数据、不隐式下载”的边界。

## 步骤 5：扩展归一化统计命令

修改 [`scripts/compute_norm_stats.py`](../../scripts/compute_norm_stats.py)。

新增 imports：

```python
import dataclasses
import pathlib
```

把 `main` 签名改为：

```python
def main(config_name: str, dataset_root: pathlib.Path | None = None, max_frames: int | None = None):
```

在获取 config 后增加：

```python
if dataset_root is not None:
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, dataset_root=str(dataset_root)))
elif isinstance(config.data, _config.LeRobotAgiBotG01DataConfig):
    raise ValueError("pi05_agibot_g01 requires --dataset-root pointing to the task_5093 directory")
```

原因：统计量必须基于 G01 变换后的 16 维 `state` 和 16 维 `actions`，而不是原始 163/36 维数组。这个脚本的变换顺序是 repack 和 robot transform 后停止，不做 normalize 和 tokenization，正好满足统计量生成需求。

## 步骤 6：新增 G01 policy transforms

新增 [`src/openpi/policies/agibot_g01_policy.py`](../../src/openpi/policies/agibot_g01_policy.py)。

### 6.1 固定维度和 delta mask

定义：

```python
RAW_STATE_DIM = 163
RAW_ACTION_DIM = 36
POLICY_DIM = 16
JOINT_ACTION_MASK = transforms.make_bool_mask(14, -2)
```

`JOINT_ACTION_MASK` 的含义是前 14 维手臂关节使用 delta action，最后 2 维夹爪保持绝对动作。

### 6.2 图像转换

`_convert_image` 要同时接受两种输入：

- 训练侧 LeRobot 解码：CHW float，数值一般在 `[0, 1]`；
- 推理侧 ROS2：HWC uint8 RGB。

输出统一为 HWC uint8 RGB。遇到非三维图像、无法判断通道轴、非 RGB、NaN 或 Inf 时直接报错。

### 6.3 state/action 切片

`_select_state` 实现：

```python
state[28:42] + state[0:2]
```

也就是：

- `state[28:35]`：左臂 7 维；
- `state[35:42]`：右臂 7 维；
- `state[0:1]`：左夹爪；
- `state[1:2]`：右夹爪。

`_select_actions` 实现：

```python
action[16:30] + action[0:2]
```

也就是：

- `action[16:23]`：左臂 7 维；
- `action[23:30]`：右臂 7 维；
- `action[0:1]`：左夹爪；
- `action[1:2]`：右夹爪。

这两个函数也接受已经是 16 维的推理侧数据，方便 ROS2 客户端直接发送策略 state。

### 6.4 输入字段映射

`AgiBotG01Inputs` 输出 openpi 模型通用字段：

| 输入字段 | 输出字段 |
| --- | --- |
| `images.top_head` | `image.base_0_rgb` |
| `images.hand_left` | `image.left_wrist_0_rgb` |
| `images.hand_right` | `image.right_wrist_0_rgb` |
| `state` | `state` |
| `actions` | `actions`，仅训练侧存在 |
| `prompt` | `prompt` |

同时写入三个 `image_mask`，均为 true。

### 6.5 输出去 padding

`AgiBotG01Outputs` 从模型返回的 `(T, 32)` action 中截取前 16 维：

```python
actions = np.asarray(data["actions"], dtype=np.float32)[..., :POLICY_DIM]
```

它要求最终 shape 为 `(T, 16)`，并拒绝 NaN/Inf。注意：绝对动作恢复由 `AbsoluteActions` 完成，`AgiBotG01Outputs` 只负责去掉模型 padding 和最终校验。

## 步骤 7：给 WebSocket 客户端增加接收超时

修改 [`packages/openpi-client/src/openpi_client/websocket_client_policy.py`](../../packages/openpi-client/src/openpi_client/websocket_client_policy.py)。

在 `WebsocketClientPolicy.__init__` 增加参数：

```python
receive_timeout: float | None = None
```

保存为：

```python
self._receive_timeout = receive_timeout
```

在 `infer` 中把：

```python
response = self._ws.recv()
```

改成：

```python
response = self._ws.recv(timeout=self._receive_timeout)
```

这不是模型能力改动，而是真机安全边界：策略服务端卡住时，ROS2 客户端不能无限阻塞。

## 步骤 8：新增 ROS2 客户端目录

新增目录 [`examples/agibot_g01`](../../examples/agibot_g01)。

### 8.1 `__init__.py`

新增 [`examples/agibot_g01/__init__.py`](../../examples/agibot_g01/__init__.py)，让目录可作为 Python package 导入。

### 8.2 `control_utils.py`

新增 [`examples/agibot_g01/control_utils.py`](../../examples/agibot_g01/control_utils.py)，只依赖 NumPy，便于在非 ROS 环境测试。

核心职责：

- `validate_server_metadata`：要求 `robot_type=agibot_g01`、`action_horizon=16`、`policy_action_dim=16`、`dataset_fps=30`。
- `parse_action_chunk`：要求返回字典包含 `actions`，shape 为 `(T, 16)`，非空且 finite。
- `resample_sequence`：把 30 Hz 动作序列线性重采样到 60 Hz。
- `prepare_execution_chunk`：默认取前 `execute_horizon=8` 步，再重采样。
- `EMAFilter`：动作平滑。
- `JointSafetyLimiter`：限制关节步长、速度和加速度。
- `blend_target`：动作块开始处从当前关节平滑融合到目标。

### 8.3 `main.py`

新增 [`examples/agibot_g01/main.py`](../../examples/agibot_g01/main.py)，实现 ROS2 真机客户端。

输入 topic 默认值：

| Topic | 消息 | 用途 |
| --- | --- | --- |
| `/camera/head_color` | `sensor_msgs/Image` | 头部 RGB |
| `/camera/hand_left_color` | `sensor_msgs/Image` | 左手 RGB |
| `/camera/hand_right_color` | `sensor_msgs/Image` | 右手 RGB |
| `/hal/arm_joint_state` | `sensor_msgs/JointState` | 前 14 个手臂关节 |
| `/hal/left_ee_data` | `genie_msgs/EndState` | 左夹爪状态 |
| `/hal/right_ee_data` | `genie_msgs/EndState` | 右夹爪状态 |

输出 topic 默认值：

| Topic | 消息 | 用途 |
| --- | --- | --- |
| `/wbc/arm_command` | `sensor_msgs/JointState` | 14 维手臂绝对位置命令 |
| `/wbc/left_ee_command` | `sensor_msgs/JointState` | 左夹爪命令 |
| `/wbc/right_ee_command` | `sensor_msgs/JointState` | 右夹爪命令 |

安全默认值：

- `enable_control=False`，也就是默认 `--no-enable-control`。
- 只有显式传 `--enable-control` 时才创建 publisher。
- 启控默认要求输入 `ENABLE`。
- 默认只执行模型输出的前 8 步。
- 默认从 `model_fps=30` 重采样到 `control_hz=60`。
- 默认 `blend_steps=14`、`max_step_delta=0.2618`、`max_joint_speed=3.0`、`max_joint_accel=6.0`。
- 左夹爪默认限制 `[0, 0.0018111112]`，右夹爪默认限制 `[0, 1.0]`。

客户端发送给策略服务器的 observation 是：

```text
images.top_head    H×W×3 uint8 RGB
images.hand_left   H×W×3 uint8 RGB
images.hand_right  H×W×3 uint8 RGB
state              float32[16]，顺序为左臂7、右臂7、左夹爪1、右夹爪1
prompt             完整任务语言
```

这与 `AgiBotG01Inputs` 的推理路径一致。

## 步骤 9：补测试

新增或修改以下测试文件。

### 9.1 `agibot_g01_policy_test.py`

新增 [`src/openpi/policies/agibot_g01_policy_test.py`](../../src/openpi/policies/agibot_g01_policy_test.py)，覆盖：

- 原始 163 维 state 切成 16 维；
- 原始 36 维 action chunk 切成 `(16,16)` 或测试中的等价 chunk；
- CHW float 图像转 HWC uint8；
- 16 维推理输入路径；
- `DeltaActions` / `AbsoluteActions` 对前 14 维往返，对夹爪保持绝对；
- `(16,32)` 模型输出去 padding 为 `(16,16)`；
- 错误 state 维度和 NaN 拒绝；
- 可选真实数据集 smoke test，依赖 `OPENPI_G01_TEST_DATASET_ROOT`。

### 9.2 `agibot_g01_client_test.py`

新增 [`src/openpi/policies/agibot_g01_client_test.py`](../../src/openpi/policies/agibot_g01_client_test.py)，覆盖：

- 服务端 metadata 不匹配；
- action response 缺字段、shape 错误、空 chunk、NaN/Inf；
- 30 Hz 到 60 Hz 重采样；
- `prepare_execution_chunk` 只取前 8 步；
- EMA 和关节限制器行为。

### 9.3 `data_loader_test.py`

扩展 [`src/openpi/training/data_loader_test.py`](../../src/openpi/training/data_loader_test.py)，覆盖：

- `dataset_root` 被传给 `LeRobotDatasetMetadata` 和 `LeRobotDataset`；
- `delta_timestamps` 使用 `dataset_meta.fps` 和 `action_horizon`；
- 指向不存在 root 时，在 LeRobot 初始化前失败；
- `local_dataset_only=True` 且未提供 root 时失败，不回退下载。

## 步骤 10：补中文文档

新增或更新：

- [`docs/PROJECT_DOCUMENTATION.md`](../PROJECT_DOCUMENTATION.md)
- [`docs/architecture.md`](../architecture.md)
- [`docs/modules/data-and-training.md`](../modules/data-and-training.md)
- [`docs/modules/inference-and-deployment.md`](../modules/inference-and-deployment.md)
- [`docs/implementation/agibot-g01.md`](agibot-g01.md)
- [`docs/principles/pi05-flow-matching.md`](../principles/pi05-flow-matching.md)
- [`docs/CHANGELOG_RECENT.md`](../CHANGELOG_RECENT.md)
- [`examples/agibot_g01/README.md`](../../examples/agibot_g01/README.md)
- 根目录 [`README.md`](../../README.md) 增加项目文档索引链接。

这一步不影响运行逻辑，但保留了当前实现的代码依据、部署命令和未验证边界。

## 迁移后的关键命令

计算归一化统计量：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_agibot_g01 \
  --dataset-root /path/to/task_5093
```

启动 JAX 全参数微调：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_agibot_g01 \
  --data.dataset-root /path/to/task_5093 \
  --exp-name g01_task_5093 \
  --overwrite
```

启动策略服务。训练 30,000 步时，最后 step 目录通常是 `29999`：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_agibot_g01 \
  --policy.dir checkpoints/pi05_agibot_g01/g01_task_5093/29999 \
  --port 8000
```

启动 ROS2 只观察客户端：

```bash
python examples/agibot_g01/main.py \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --no-enable-control
```

只有完成离线检查、只观察检查和低速短周期验收后，才启用真机控制：

```bash
python examples/agibot_g01/main.py \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --enable-control
```

## 验证顺序

从官方代码迁移后，建议按以下顺序验证：

1. 语法检查：

   ```bash
   python3 -m compileall \
     src/openpi/policies/agibot_g01_policy.py \
     examples/agibot_g01 \
     scripts/compute_norm_stats.py
   ```

2. 运行轻量测试：

   ```bash
   pytest \
     src/openpi/policies/agibot_g01_policy_test.py \
     src/openpi/policies/agibot_g01_client_test.py \
     src/openpi/training/data_loader_test.py
   ```

3. 如果本机有真实数据集，运行真实样本 smoke test：

   ```bash
   OPENPI_G01_TEST_DATASET_ROOT=/path/to/task_5093 \
     pytest src/openpi/policies/agibot_g01_policy_test.py -m manual
   ```

4. 计算统计量后检查生成的 assets 只包含变换后的 `state` 和 `actions` 统计量。

5. 启动策略服务器，用 ROS2 客户端 `--no-enable-control --max-cycles 10` 检查动作 shape、数值范围和服务端 metadata。

6. 真机验收顺序必须是：

   ```text
   默认禁控观察 → 保存动作并人工检查 → 低速短周期启控 → 放宽周期和速度参数
   ```

## 常见合并错误

- 只注册 `pi05_agibot_g01`，但忘记导入 `agibot_g01_policy`：配置创建会失败。
- 只给 `DataConfig` 加 `dataset_root`，但忘记给 `DataConfigFactory` 加同名字段：CLI 的 `--data.dataset-root` 不会生效。
- 忘记在 LeRobot metadata 和 dataset 两处都传 `root`：metadata 可能仍尝试按 repo id 解析远程数据。
- 用 `action_dim=16` 直接训练 π0.5：会绕开当前实现的 32 维模型 padding 约定，与 `PadStatesAndActions` 和基础权重形状不匹配。
- 把夹爪也纳入 delta mask：会改变已经确认可直接执行的夹爪 action 单位。
- 在 ROS2 客户端默认启用 publisher：不符合当前安全边界。
- 使用 `30000` 作为最终检查点目录：当前训练循环的最后 step index 是 `29999`。

## 当前环境中已验证与未验证

已验证：

- Python 语法检查曾在变更文件上通过。
- `git diff --check` 曾通过。
- 纯 NumPy 控制工具冒烟检查曾通过。
- Markdown 链接校验曾通过。

未在当前 Mac 环境完整验证：

- LeRobot 视频解码和真实 `task_5093` 数据加载。
- JAX π0.5 GPU 训练。
- 策略服务器加载真实训练检查点。
- ROS2 与真机 topic 对接。
- 真机控制执行。
