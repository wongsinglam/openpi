"""Custom train configs loaded by `openpi.training.config` at runtime.

This file is intentionally outside `src/openpi/...` so local extensions can live
in one place and stay easy to maintain during upstream rebases.
"""

from openpi.models import pi0_config
from openpi.training.config import DataConfig
from openpi.training.config import FakeDataConfig
from openpi.training.config import TrainConfig
from openpi.training import weight_loaders

from openpi_ext.training.data_configs import *


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


_EXT_CONFIGS = [
    TrainConfig(
        name="pi05_all_action",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotAgibotDataConfig(
            repo_id="Lerobot_all_action_30",  # ✅ 简单的标识符，不是路径
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_agibot_eraser_action",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotAgibotDataConfig(
            repo_id="Lerobot_Eraser_action",  # ✅ 简单的标识符，不是路径
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_agibot_glue_action",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotAgibotDataConfig(
            repo_id="Lerobot_Glue_action",  # ✅ 简单的标识符，不是路径
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_agibot_pen_action",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotAgibotDataConfig(
            repo_id="Lerobot_Pen_action",  # ✅ 简单的标识符，不是路径
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_agibot_stickynote_action",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotAgibotDataConfig(
            repo_id="Lerobot_StickyNote_action",  # ✅ 简单的标识符，不是路径
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi05_agibot_test_glue_action",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotAgibotDataConfig(
            repo_id="Lerobot_Test_Glue_action",  # ✅ 简单的标识符，不是路径
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_agibot_wipe_action",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotAgibotDataConfig(
            repo_id="Lerobot_Wipe_action",  # ✅ 简单的标识符，不是路径
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi05_agibot_glue_best_action",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotAgibotDataConfig(
            repo_id="Lerobot_Glue_best",  # ✅ 简单的标识符，不是路径
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_box_100",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotFrankaDataConfig(
            repo_id="box_100_joints_1028",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=10_000,  # 数据更多，训练更久
        batch_size=16,
    ),

    TrainConfig(
        name="pi05_trash_201",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotFrankaDataConfig(
            repo_id="trash_201_joints_1028",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,  # 201 episodes, 97086帧 - 训练更久
        batch_size=16,
    ),

    #
    # Franka EE (End-Effector) space configs - using absolute actions
    #
    TrainConfig(
        name="pi0_box_100_ee",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotFrankaEEDataConfig(
            repo_id="box_100_ee_1028",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,  # 数据更多，训练更久
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_box_100_ee",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotFrankaEEDataConfig(
            repo_id="box_100_ee_1028",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,  # 100 episodes, 34035 frames - EE space
        batch_size=16,
    ),
    TrainConfig(
        name="pi0_trash_201_ee",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotFrankaEEDataConfig(
            repo_id="trash_201_ee_1028",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,  # 数据更多，训练更久
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_trash_201_ee",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotFrankaEEDataConfig(
            repo_id="trash_201_ee_1028",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,  # 201 episodes, ~97k frames - EE space
        batch_size=16,
    ),
    TrainConfig(
    name="pi0_fruits_50_ee",
    model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
    data=LeRobotFrankaEEDataConfig(
        repo_id="fruits_50_ee",
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,  # 数据更多，训练更久
    batch_size=16,
    ),
    TrainConfig(
        name="pi05_fruits_50_ee",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotFrankaEEDataConfig(
            repo_id="fruits_50_ee",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,  # 50 episodes, 9725 frames - EE space
        batch_size=16,
    ),
    TrainConfig(
    name="pi0_bread_100_ee",
    model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
    data=LeRobotFrankaEEDataConfig(
        repo_id="bread_100_ee_1028",
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,  # 数据更多，训练更久
    batch_size=16,
    ),
    TrainConfig(
        name="pi05_bread_100_ee",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotFrankaEEDataConfig(
            repo_id="bread_100_ee_1028",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,  # 100 episodes, 51419 frames - EE space
        batch_size=16,
    ),

    TrainConfig(
        name="pi0_fruits_160_ee",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotFrankaEEDataConfig(
            repo_id="fruits_160_ee",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,  # 50 episodes, 9725 frames - EE space
        batch_size=16,
    ),

    TrainConfig(
        name="pi05_fruits_160_ee",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotFrankaEEDataConfig(
            repo_id="fruits_160_ee",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,  # 50 episodes, 9725 frames - EE space
        batch_size=16,
    ),

    TrainConfig(
        name="pi0_bowls_120_ee",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotFrankaEEDataConfig(
            repo_id="bowls_120_ee",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,  # 50 episodes, 9725 frames - EE space
        batch_size=16,
    ),


    TrainConfig(
        name="pi05_bowls_120_ee",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotFrankaEEDataConfig(
            repo_id="bowls_120_ee",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,  # 50 episodes, 9725 frames - EE space
        batch_size=16,
    ),

    #
    # Dobot dual-arm robot configs
    #
    # Cook Vegetable
    TrainConfig(
        name="pi0_dobot_cook_vegetable",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotDobotDataConfig(
            repo_id="dobot_cook_vegetable",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=35_000,  # 201 episodes, 83215 frames
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_dobot_cook_vegetable",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotDobotDataConfig(
            repo_id="dobot_cook_vegetable",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=35_000,  # 201 episodes, 83215 frames
        batch_size=16,
    ),

    # Pour Water
    TrainConfig(
        name="pi0_dobot_pour_water",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotDobotDataConfig(
            repo_id="dobot_pour_water",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=35_000,  # 203 episodes, 61334 frames
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_dobot_pour_water",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotDobotDataConfig(
            repo_id="dobot_pour_water",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=35_000,  # 203 episodes, 61334 frames
        batch_size=16,
    ),

    # Tidy Up Desk
    TrainConfig(
        name="pi0_dobot_tidy_desk",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotDobotDataConfig(
            repo_id="dobot_tidy_up_the_desk_full",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=35_000,  # 201 episodes, 59300 frames
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_dobot_tidy_desk",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotDobotDataConfig(
            repo_id="dobot_tidy_up_the_desk",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=35_000,  # 201 episodes, 59300 frames
        batch_size=16,
    ),

    # Towel Collect
    TrainConfig(
        name="pi0_dobot_towel_collect",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotDobotDataConfig(
            repo_id="dobot_towel_collect",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=35_000,  # 203 episodes, 90663 frames
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_dobot_towel_collect",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotDobotDataConfig(
            repo_id="dobot_towel_collect",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=35_000,  # 203 episodes, 90663 frames
        batch_size=16,
    ),

    #
    # Dobot dual-arm robot Full datasets
    #
    # Cook Vegetable Full
    TrainConfig(
        name="pi0_dobot_cook_vegetable_full",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_cook_vegetable_full",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,  # 201 episodes, 83215 frames
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_dobot_cook_vegetable_full",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_cook_vegetable_full",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,  # 201 episodes, 83215 frames
        batch_size=16,
    ),

    # Pour Water Full
    TrainConfig(
        name="pi0_dobot_pour_water_full",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_pour_water_full",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,  # 203 episodes, 61334 frames
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_dobot_pour_water_full",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_pour_water_full",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,  # 203 episodes, 61334 frames
        batch_size=16,
    ),

    # Tidy Up Desk Full
    TrainConfig(
        name="pi0_dobot_tidy_desk_full",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_tidy_up_the_desk_full",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,  # 201 episodes, 59300 frames
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_dobot_tidy_desk_full",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_tidy_up_the_desk_full",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,  # 201 episodes, 59300 frames
        batch_size=16,
    ),

    # Towel Full
    TrainConfig(
        name="pi0_dobot_towel_full",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_towel_full",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,  # 203 episodes, 90663 frames
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_dobot_towel_full",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_towel_full",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,  # 203 episodes, 90663 frames
        batch_size=16,
    ),
    # Stack Bowls Full
    TrainConfig(
        name="pi05_dobot_stack_bowls",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_stack_bowls",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,  # update if you want to scale with dataset size
        batch_size=16,
    ),
    #
    # Dobot LeRobot crop datasets
    #
    # Grab Pass Place
    TrainConfig(
        name="pi05_dobot_grab_pass_place",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_crop_grab_pass_place",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=35_000,  # 70 episodes, 19816 frames
        batch_size=16,
    ),

    # Dobot Crop datasets - π₀ models
    # Grab Pass Place (crop version)
    TrainConfig(
        name="pi0_dobot_crop_grab_pass_place",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_crop_grab_pass_place",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,  # 70 episodes, 19816 frames
        batch_size=16,  # Divisible by 3 GPUs
    ),

    # Stack Cups (crop version)
    TrainConfig(
        name="pi0_dobot_crop_stack_cups",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_crop_stack_cups",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=35_000,  # 103 episodes, 40059 frames
        batch_size=16,  # Divisible by 3 GPUs
    ),

    # Tidy Desk (crop version)
    TrainConfig(
        name="pi0_dobot_crop_tidy_desk",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_crop_tidy_desk",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,  # 51 episodes, 8909 frames
        batch_size=16,  # Divisible by 3 GPUs
    ),

    # Pealing Fruit
    TrainConfig(
        name="pi0_dobot_pealing_fruit",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_pealing_fruit",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=35_000,  # 103 episodes, 71544 frames
        batch_size=16,  # Divisible by 3 GPUs
    ),

    # Dobot Crop datasets - π₀.₅ models
    # Grab Pass Place (crop version)
    TrainConfig(
        name="pi05_dobot_crop_grab_pass_place",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_crop_grab_pass_place",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,  # 70 episodes, 19816 frames
        batch_size=16,  # Divisible by 3 GPUs
    ),

    # Stack Cups
    TrainConfig(
        name="pi05_dobot_crop_stack_cups",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_crop_stack_cups",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=35_000,  # 103 episodes, 40059 frames
        batch_size=16,  # Divisible by 3 GPUs
    ),

    # Tidy Desk (crop version)
    TrainConfig(
        name="pi05_dobot_crop_tidy_desk",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_crop_tidy_desk",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,  # 51 episodes, 8909 frames
        batch_size=16,  # Divisible by 3 GPUs
    ),

    # Pealing Fruit
    TrainConfig(
        name="pi05_dobot_pealing_fruit",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotDobotFullDataConfig(
            repo_id="dobot_pealing_fruit",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=35_000,  # 103 episodes, 71544 frames
        batch_size=16,  # Divisible by 3 GPUs
    ),

    TrainConfig(
        name="pi0_blocks_99_ee",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotFrankaEEDataConfig(
            repo_id="blocks_99_ee",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,  # 50 episodes, 9725 frames - EE space
        batch_size=16,
    ),

    TrainConfig(
        name="pi05_blocks_99_ee",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotFrankaEEDataConfig(
            repo_id="blocks_99_ee",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,  # 50 episodes, 9725 frames - EE space
        batch_size=16,
    ),

    TrainConfig(
        name="pi0_cup_130_ee",
        model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
        data=LeRobotFrankaEEDataConfig(
            repo_id="cup_130_ee",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,  # 130 episodes, 29696 frames - EE space
        batch_size=16,  # Divisible by 3 GPUs
    ),

    TrainConfig(
        name="pi05_cup_130_ee",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotFrankaEEDataConfig(
            repo_id="cup_130_ee",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,  # 130 episodes, 29696 frames - EE space
        batch_size=16,  # Divisible by 3 GPUs
    ),
]


def get_configs() -> list[TrainConfig]:
    """Return all extension configs for `python3 -m openpi_ext.train`."""
    return [
        EXT_DEBUG_CONFIG,
        DOBOT_TIDY_DESK_FULL_CONFIG,
        *_EXT_CONFIGS,
    ]

