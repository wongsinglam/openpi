# openpi_ext 变更记录

日期：2026-02-06

本文档汇总 `openpi copy` 与 `openpi` 原版的关键差异。

## droid_rlds_dataset.py

对比文件：
- 修改版：`openpi copy/src/openpi/training/droid_rlds_dataset.py`
- 原版：`openpi/src/openpi/training/droid_rlds_dataset.py`

### 主要改动

- 移除多数据集采样机制
  - 删除 `RLDSDataset` dataclass 和 `datasets: Sequence[RLDSDataset]` 参数
  - 不再 `sample_from_datasets`，改为单一 droid 数据集

- 数据集构建方式简化为固定 droid
  - 原版：对 `datasets` 逐个 `prepare_single_dataset(...)`
  - 修改版：直接 `tfds.builder("droid", version="1.0.1")`

- 过滤字典改为单一可选参数
  - 原版：`RLDSDataset.filter_dict_path` 可为每个子数据集单独设置
  - 修改版：`filter_dict_path` 作为 `DroidRldsDataset.__init__` 参数

- 管线逻辑挪到主流程
  - 原版把 `restructure / chunk_actions / flatten / filter / decode` 封装在 `prepare_single_dataset`
  - 修改版直接作用于单一 `dataset`

- 日志与变量命名变化
  - 原版使用 `final_dataset` 并输出多数据集准备日志
  - 修改版只处理 `dataset` 并直接赋值 `self.dataset = dataset`

### 行为保持不变（仍保留）

- 成功轨迹过滤（正则匹配 `file_path`）
- action chunking
- filter dict hash table 过滤
- 解码 `image` / `wrist_image`
- `shuffle` + `batch` + `with_ram_budget(1)`


## data_loader.py

对比文件：
- 修改版：`openpi copy/src/openpi/training/data_loader.py`
- 原版：`openpi/src/openpi/training/data_loader.py`

### 主要改动

#### 1) LeRobot 数据集加载改为支持本地离线数据

- 强制离线：设置 `HF_HUB_OFFLINE=1`
- 支持通过 `HF_LEROBOT_HOME` 指定本地数据集根目录
  - 若未设置，默认 `~/.cache/huggingface/lerobot`
- monkey patch `lerobot.common.datasets.utils.get_repo_versions`
  - 从本地 `meta/info.json` 读取 `codebase_version`
  - 避免访问 HuggingFace Hub
- 若本地数据存在，读取 `revision=codebase_version` 并提示；否则尝试在线下载
- 构建 `LeRobotDatasetMetadata` / `LeRobotDataset` 时：
  - 不传 `root` 参数，使用 `HF_LEROBOT_HOME / repo_id`
  - 设置 `tolerance_s=200.0`
  - 设置 `video_backend="pyav"`

#### 2) 移除字符串列，避免 JAX 报错

- 如果 `dataset.hf_dataset` 有字符串特征：
  - 移除 `task`
  - 移除 `observation.images.*`（路径字符串列）

#### 3) RLDS Loader 参数变化

- 调用 `DroidRldsDataset` 时，从 `datasets=data_config.datasets`
  改为 `filter_dict_path=data_config.filter_dict_path`

#### 4) DataLoader 迭代时过滤字符串/bytes

- 在 `__iter__` 中加入过滤：
  - 剔除 string/bytes/含字符串 dtype 的 numpy 数组
  - 递归移除 None 节点

#### 5) `_collate_fn` 在 stack 前过滤字符串字段

- 对每个 item：剔除值为 string 或 list[str] 的字段
- 再做 `np.stack`

### 行为保持不变（仍保留）

- FakeDataset 分支未变
- RLDS pipeline 构建及 batching 结构保持
- 训练中 batch 输出仍按原流程进行 sharding / torch tensor 转换
