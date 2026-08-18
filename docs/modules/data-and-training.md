# 数据与训练

## 数据配置

`DataConfigFactory.create` 生成不可变的 `DataConfig`，其中包含数据集标识、归一化统计量、变换组、动作序列字段和可选 RLDS 配置。`create_base_config` 为 π0.5 和 π0-FAST 选择分位数归一化，为 π0 选择 z-score 归一化。

`create_torch_dataset` 负责打开 LeRobot 数据集。当前工作区新增 `dataset_root`，用于直接加载本地数据；同时新增 `local_dataset_only`，用于禁止特定配置回退到 Hub 下载。显式指定的本地根目录必须包含 `meta/info.json`。G01 工厂设置 `local_dataset_only=True`。

## 变换顺序

训练按以下顺序应用变换：

1. 可选的 `PromptFromLeRobotTask` 包装器；
2. 仅用于数据集记录的 `repack_transforms.inputs`；
3. 与推理共享的 `data_transforms.inputs`；
4. 使用机器人变换后统计量的 `Normalize`；
5. 用于 prompt 注入/tokenization、图像缩放和维度填充的 `model_transforms.inputs`。

推理使用相同的机器人输入变换和模型输入变换，但跳过仅训练使用的 repack。输出处理依次应用模型输出处理、反归一化、机器人输出变换，以及可选的调用方 repack。

`Group.push` 会将新的输入变换追加到末尾，但将输出变换插入开头。因此，delta 动作转换可以在机器人输入变换后添加，而其逆变换会在机器人输出变换前执行。

## 通用数据结构

变换必须生成：

- `image`：三个模型图像字段，loader 组成 batch 前均为无 batch 维的 HWC；
- `image_mask`：每个图像字段对应一个布尔值；
- `state`：`float32[..., state_dim]`；
- 模型变换后的 prompt token 数组；
- 训练期间的 `actions`：`float32[..., action_horizon, action_dim]`。

`Observation.from_dict` 将该字典转换为有类型的模型结构，并将 `[0,255]` 的 uint8 图像转换为 `[-1,1]` 的 float32。`preprocess_observation` 在需要时将图像缩放到 224×224，并应用训练数据增强。

## 归一化

`scripts/compute_norm_stats.py` 构造同一数据集，执行 repack 和机器人输入变换，并刻意停在归一化与模型 tokenization 之前。它只累计 `state` 和 `actions` 的统计量，并将其写入 `config.assets_dirs / repo_id`。

对于 π0.5，`Normalize` 将 q01–q99 区间近似映射到 `[-1,1]`，`Unnormalize` 执行逆映射。统计量会复制到每个 JAX 检查点的 `assets/<asset_id>` 下；服务端加载检查点内的副本，避免训练配置漂移。复用预训练 assets 的方法见[归一化统计量](../norm_stats.md)。

## JAX 训练生命周期

`scripts/train.py` 执行以下操作：

- 校验全局 batch size 能被 JAX 设备数整除；
- 创建 `(batch, fsdp)` 设备 mesh；
- 初始化检查点管理器和 W&B run；
- 加载一批数据并记录相机视图；
- 初始化模型并合并基础检查点权重；
- 使用明确的数据和状态 sharding 对 `train_step` 进行 JIT 编译；
- 计算梯度、应用 Optax AdamW、更新可选 EMA、记录指标并保存检查点。

默认 `CosineDecaySchedule` 预热 1,000 步，在 `2.5e-5` 达到峰值，并在 30,000 步内衰减到 `2.5e-6`。具体配置可以覆盖这些默认值。全参数微调使用 `nnx.Nothing` 作为冻结过滤器；LoRA 配置通过 `Pi0Config.get_freeze_filter` 构造变体和冻结过滤器。

## 分片与检查点

`make_mesh(fsdp_devices)` 要求设备总数能被 `fsdp_devices` 整除。`fsdp_sharding` 会复制标量、向量和小于 4 MiB 的张量；更大的数组沿可被 FSDP 轴整除的最大维度进行分片。数据跨 mesh 的两个轴分片。

JAX 检查点包含 `train_state`、推理用 `params` 和 assets。启用 EMA 时，`_split_params` 将 EMA 权重保存为推理参数。`overwrite` 会删除已有实验目录，`resume` 会恢复最后一个可用训练状态；两者不能同时设置。

`scripts/train_pytorch.py` 中的 PyTorch 路径复用数据配置，但使用 DDP，并写入 `model.safetensors`、`optimizer.pt`、`metadata.pt` 和 assets。它是独立执行路径，不是 JAX trainer 内部的切换选项。

## 失败模式

- 缺少归一化 assets 时，训练会在 `transform_dataset` 中失败。
- 指定 `dataset_root` 后，错误的本地数据根目录会在 LeRobot 尝试远程下载之前失败。
- 未列入 `action_sequence_keys` 的动作字段不会进行 horizon 采样。
- 错误的机器人变换可能生成形状合法但语义错误的数据；仅靠 shape 测试无法发现。
- batch size 或 FSDP 整除错误会在训练开始前终止。
- 当前文档环境未执行 LeRobot 解码或 GPU 训练。
