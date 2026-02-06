"""Extension training entrypoint without modifying core OpenPI config registry."""

import argparse
import importlib.util
import pathlib
import sys
import types

import tyro

from openpi.training.config import TrainConfig
import openpi.training.config as core_config
from openpi_ext.training.configs import get_configs
from openpi_ext.training import data_loader_ext


def _load_base_train_module() -> types.ModuleType:
    """Load `scripts/train.py` as a Python module."""
    try:
        from scripts import train as train_module

        return train_module
    except Exception:
        pass

    train_path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_scripts_train", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load train module from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_configs() -> dict[str, TrainConfig]:
    configs = dict(core_config._CONFIGS_DICT)
    ext_configs = {cfg.name: cfg for cfg in get_configs()}
    duplicate_names = set(configs).intersection(ext_configs)
    if duplicate_names:
        dup = ", ".join(sorted(duplicate_names))
        raise ValueError(f"Extension config names must be unique and not shadow core configs: {dup}")
    configs.update(ext_configs)
    return configs


def cli() -> TrainConfig:
    configs = _all_configs()
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in configs.items()})


def _parse_args(argv: list[str]) -> tuple[bool, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--use-ext-loader", action="store_true")
    args, rest = parser.parse_known_args(argv)
    return args.use_ext_loader, rest


def main() -> None:
    train_module = _load_base_train_module()
    use_ext_loader, rest = _parse_args(sys.argv[1:])
    if use_ext_loader:
        # Swap in extension data loader without touching core code.
        train_module._data_loader = data_loader_ext
    sys.argv[1:] = rest
    train_module.main(cli())


if __name__ == "__main__":
    main()
