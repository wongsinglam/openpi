"""Custom train configs loaded by `openpi.training.config` at runtime.

This file is intentionally outside `src/openpi/...` so local extensions can live
in one place and stay easy to maintain during upstream rebases.
"""

from openpi.models import pi0_config
from openpi.training.config import DataConfig
from openpi.training.config import FakeDataConfig
from openpi.training.config import TrainConfig
from openpi.training import weight_loaders

from openpi_ext.data_configs import LeRobotDobotFullDataConfig


EXT_DEBUG_CONFIG = TrainConfig(
    name="ext_debug",
    exp_name="ext_debug",
    data=FakeDataConfig(),
    batch_size=2,
    model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
    save_interval=100,
    overwrite=True,
    num_train_steps=10,
    wandb_enabled=False,
)

DOBOT_TIDY_DESK_FULL_CONFIG = TrainConfig(
    name="pi05_dobot_tidy_desk_full",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=50,
    ),
    data=LeRobotDobotFullDataConfig(
        repo_id="dobot_tidy_up_the_desk_full",
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=30_000,
    batch_size=16,
)


def get_configs() -> list[TrainConfig]:
    """Return all extension configs that should be merged into `_CONFIGS`."""
    return [
        EXT_DEBUG_CONFIG,
        DOBOT_TIDY_DESK_FULL_CONFIG,
    ]
