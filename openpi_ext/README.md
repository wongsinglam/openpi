# openpi_ext

本目录用于放置你自己的扩展代码（尤其是训练 config），避免直接改动主干配置列表。

## 当前接入方式

- 使用扩展入口 `openpi_ext/train.py`，它会：
  - 读取核心配置（`openpi.training.config`）
  - 读取扩展配置（`openpi_ext.configs:get_configs()`）
  - 合并后二次构建 CLI（不改动 core 代码）

```bash
python3 -m openpi_ext.train --help
python3 -m openpi_ext.train pi05_dobot_tidy_desk_full --exp_name your_exp
```

> 具体参数以 `python3 -m openpi_ext.train --help` 为准。

## 推荐扩展结构

- `openpi_ext/configs.py`：维护配置注册（`get_configs()`）
- `openpi_ext/data_configs.py`：自定义 DataConfigFactory
- `openpi_ext/train.py`：扩展训练入口（合并 core + ext configs）
- `openpi_ext/datasets/`：自定义数据处理/转换
- `openpi_ext/policies/`：自定义输入输出 transform 或 policy glue
- `openpi_ext/utils/`：扩展工具函数

你后续只要在 `get_configs()` 里新增或导入配置即可。
