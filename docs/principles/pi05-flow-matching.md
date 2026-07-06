# 本仓库中的 π0.5 流匹配

## 问题定义

π0.5 根据图像、语言和机器人 state，预测固定 horizon 的连续动作轨迹。它与 π0 使用相同的 `Pi0` 类，通过 `Pi0Config(pi05=True)` 选择。该配置改变 state 条件输入和时间步条件输入，同时保留流匹配动作头。

设 `a` 为归一化后的动作块，`ε` 为标准高斯噪声。`Pi0.compute_loss` 采样 `t ~ Beta(1.5,1)` 并将其限制在 `(0,1]`，随后构造：

```text
x_t = t ε + (1 - t) a
u_t = ε - a
```

网络预测速度 `v_t(x_t, observation, t)`。训练时在动作维度上最小化均方误差：

```text
L = mean_dim((v_t - u_t)²)
```

该方法为动作序列中的每个时间步返回损失；JAX trainer 在求导前将其平均为一个标量。

## Observation 条件输入

prefix 由 SigLIP 图像 token 和随后的 PaliGemma prompt token 组成。图像和语言 prefix token 使用双向注意力。action suffix 包含投影后的带噪动作 token，并对 prefix 执行注意力。

π0.5 与 π0 在代码层面有两点差异：

1. `TokenizePrompt(..., discrete_state_input=True)` 将每个归一化 state 分量量化为 `[-1,1]` 区间内的 256 个 bin，并将生成的整数嵌入语言 prompt。π0 则在 action expert 中投影一个连续 state token。
2. π0.5 将时间步 MLP 输出作为自适应 RMS 归一化条件 `adarms_cond` 传给 action expert。π0 会把时间步 embedding 与动作 token 拼接，再通过独立 MLP 处理。

因此，π0.5 的 state 归一化是 token 语义的一部分：即使张量 shape 合法，过时或不兼容的 q01/q99 统计量也会改变离散 state 符号。

## 采样

`Pi0.sample_actions` 从 shape 为 `(batch, action_horizon, action_dim)` 的高斯噪声开始。它只计算一次图像与语言 prefix 的 key/value 并保存 KV cache，然后使用 Euler 积分从 `t=1` 向 `t=0` 推进：

```text
dt = -1 / num_steps
x ← x + dt · v_t(x, observation, t)
t ← t + dt
```

默认执行十个积分步骤。返回的 `x_0` 仍位于归一化后的模型动作空间；策略输出变换负责反归一化，并恢复机器人所需的绝对动作表示。

## 维度填充

`Pi0Config.inputs_spec` 将 state 和 action shape 固定为 `action_dim`。物理维度较少的机器人策略可以使用 `PadStatesAndActions` 在末尾补零。输出变换必须在采样后移除 padding。padding 只影响结构；归一化统计量仅覆盖物理维度，归一化辅助函数用中性默认值保留额外维度。

## 权衡与失败模式

- 流匹配一次生成完整动作块，能够降低策略查询频率，但会增加开环风险。
- 更多积分步骤会增加推理计算量；过少步骤会使数值轨迹更粗糙。
- 分位数归一化限制异常值影响，但会将 q01/q99 之外的值裁剪到模型区间。
- 错误的 absolute/delta 转换会改变目标向量场，无法在部署阶段修复。
- 错误的 padding 顺序可能让训练正常进行，却把预测映射到错误执行器。
- 图像、prompt、state 和 action 约定必须在统计量计算、训练和服务阶段保持一致。
