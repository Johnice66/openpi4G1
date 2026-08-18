# 推理与部署

## 策略构造

`scripts/serve_policy.py` 接受预定义环境或显式的 `policy:checkpoint` 组合。对于显式检查点，它会解析命名 `TrainConfig` 并调用 `create_trained_policy`。

`create_trained_policy` 执行以下步骤：

1. 解析本地或可下载的检查点路径；
2. 通过 `model.safetensors` 判断 PyTorch 格式，否则恢复 JAX `params`；
3. 重建数据配置，并从检查点加载归一化 assets；
4. 组装输入和输出变换；
5. 返回带可选元数据和设备选择的 `Policy`。

当数据配置包含 `asset_id` 时，检查点中的归一化 assets 是必需的。配置与检查点不匹配时，错误发生在参数树或 state dictionary 加载阶段，而不是 WebSocket 层。

## 策略推理

`Policy.infer` 是处理单条 observation 的字典接口。它浅复制数据树，变换 observation，添加 batch 维，转换为 JAX 数组或 PyTorch 张量，然后调用 `sample_actions`。输出变换前会移除 batch 维。返回字典包含动作以及 `policy_timing.infer_ms`。

JAX 路径中的 `sample_actions` 由 `nnx_utils.module_jit` 包装。PyTorch 路径会将模型移到选定设备并切换到 evaluation mode。接口可接受外部提供的 noise，以实现可复现采样。

## WebSocket 协议

`WebsocketPolicyServer` 监听 `0.0.0.0`，将元数据作为第一条 msgpack 消息发送，随后接收含 NumPy 数组的 msgpack observation 字典。它返回动作字典或文本 traceback。客户端禁用 WebSocket 压缩并允许无限消息大小，因为图像 observation 可能很大。

当前工作区为 `WebsocketClientPolicy` 新增 `receive_timeout`，并将其传给 `recv`，使机器人客户端在推理无响应时能够停止，而不是无限等待。连接被拒绝时仍每五秒重试一次，建立连接过程没有总超时。

通用客户端模式见[远程推理](../remote_inference.md)。

## 客户端运行时边界

`openpi-client` 刻意保持比训练包更小的依赖集合。其 `Runtime` 抽象围绕 `Environment`、`Agent` 和可选 subscriber 循环运行。`PolicyAgent` 将任意 `BasePolicy` 适配到该运行时。机器人专用 ROS 或硬件依赖保留在示例或下游项目中。

G01 客户端未使用通用 `Runtime`；它直接管理 ROS2 订阅和受保护的命令发布，因为需要协调三路相机、关节与夹爪状态、动作块重采样及安全限制。详见 [AgiBot G01 实现](../implementation/agibot-g01.md)。

## 部署命令

通用检查点服务命令：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=<config-name> \
  --policy.dir=<checkpoint-step-directory> \
  --port=8000
```

客户端必须发送所选机器人变换期望的精确推理字段。通用传输层不负责在数据集字段、环境字段或 ROS topic 之间转换。

## 故障与安全边界

- 文本形式的 WebSocket 响应会被客户端视为服务端错误。
- 非法 shape 可能在机器人变换、`Observation.from_dict`、模型类型检查或设备转换阶段失败。
- 传输元数据只是提示信息，除非客户端显式校验。
- 模型推理不强制执行机器人关节、速度、碰撞或工作空间限制。
- 原生策略服务器不校验身份认证。`WebsocketClientPolicy.api_key` 只会向兼容的代理或自定义服务器发送 Authorization header；它本身不能保护 `WebsocketPolicyServer`。
- 真机碰撞保护和急停能力不属于 openpi 的职责范围。
