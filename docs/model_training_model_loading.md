# 训练中模型调用与加载说明（基于 train.py 与 config）

本文基于以下文件整理：
- `openpi_ext/train.py`
- `scripts/train.py`
- `openpi_ext/training/configs.py`
- `src/openpi/models/pi0_config.py`
- `src/openpi/models/model.py`

## 训练中如何调用模型

训练入口是 `openpi_ext/train.py`，它做了两件关键事情：
1. **加载基础训练脚本**：优先尝试导入 `scripts/train.py`（如果存在 `openpi copy` 也会优先从那里加载）。
2. **合并并解析训练配置**：将核心配置和扩展配置合并后，通过 `tyro` CLI 选择具体 `TrainConfig`。

最终它调用：
- `train_module.main(cli())`

在 `scripts/train.py` 的 `main(config)` 中：
- `config.model.create(model_rng)` **创建模型实例**
- `train_step` 中通过 `nnx.merge(state.model_def, state.params)` **恢复模型并前向/反向**

核心调用链（简化）：
- `openpi_ext/train.py` → `scripts/train.py:main(config)`
- `init_train_state` → `config.model.create(...)`
- `train_step` → `model.compute_loss(...)`

## 模型是如何加载的

模型参数加载发生在 `scripts/train.py:init_train_state`：
1. **先按配置构建模型结构**：
   - `model = config.model.create(model_rng)`
2. **如果不是 resume**，会加载预训练权重：
   - `partial_params = _load_weights_and_validate(config.weight_loader, ...)`
   - 使用 `WeightLoader` 根据当前模型的参数形状加载“匹配的子集”
3. **把加载到的权重合并进模型**：
   - `state.replace_by_pure_dict(partial_params)`
4. **最终形成 TrainState**：
   - `nnx.state(model)` 作为 `params`

`openpi_ext/training/configs.py` 中的配置指定了：
- `model=pi0_config.Pi0Config(...)`
- `weight_loader=weight_loaders.CheckpointWeightLoader("gs://...")`

这意味着：
- **模型结构由 `Pi0Config` 决定**
- **加载的权重由 `weight_loader` 决定**

## 如果要修改模型，应如何修改

通常有三类修改方式：

### 1) 只改模型超参数（推荐）
适用于不改模型代码，只改维度、变体等。
- 在 `openpi_ext/training/configs.py` 里修改对应 `TrainConfig` 的 `model=Pi0Config(...)` 参数。
- 常见项：`action_dim`、`action_horizon`、`paligemma_variant`、`action_expert_variant`、`pi05`。

### 2) 修改模型结构代码
适用于需要改网络结构或损失。
- 修改模型实现（例如 `src/openpi/models/pi0.py`）。
- 对应的配置类在 `src/openpi/models/pi0_config.py`：
  - `create()` 必须返回更新后的模型类实例
  - `inputs_spec()` 必须与新模型期望输入一致

注意：结构变化后，已有 checkpoint 可能无法直接加载（shape 不匹配）。

### 3) 新增一个模型类型
适用于需要并行保留旧模型和新模型。
- 新建配置类（继承 `BaseModelConfig`）
- 新建模型类（继承 `BaseModel`）
- 在训练配置里改为 `model=YourNewConfig(...)`

## 修改后常见注意点
- **权重加载**：
  - 如果变更了参数形状，`CheckpointWeightLoader` 会报 shape mismatch。
  - 可以先关闭预训练加载（去掉 `weight_loader`）或提供新权重。
- **输入规格**：
  - `inputs_spec()` 需要匹配训练数据的 shape。
- **冻结策略**：
  - `Pi0Config.get_freeze_filter()` 会影响哪些参数可训练。

