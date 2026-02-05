"""Extension data-config definitions."""

import dataclasses

"""See _CONFIGS for the list of available configs."""

from collections.abc import Sequence
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi_ext.policies.agibot_policy as agibot_policy
import openpi_ext.policies.franka_policy as franka_policy
import openpi_ext.policies.dobot_policy as dobot_policy
import openpi.transforms as _transforms
from openpi.training.config import DataConfig
from openpi.training.config import DataConfigFactory
from openpi.training.config import ModelTransformFactory

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter
ToolYMode: TypeAlias = Literal["zero", "claw", "+claw", "-claw"]


@dataclasses.dataclass(frozen=True)
class LeRobotAgibotDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/head_image": "image",
                        "observation/left_image": "hand_left_image",
                        "observation/right_image": "hand_right_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[agibot_policy.AgibotInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[agibot_policy.AgibotOutputs()],
        )
        # Use delta actions (not for gripper)
        # delta_action_mask = _transforms.make_bool_mask(12, -1)
        delta_action_mask = _transforms.make_bool_mask(14, -2)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotFrankaDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
         # Repack transforms to standardize field names from LeRobot dataset
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.images.cam_high": "observation.images.cam_high",
                        "observation.images.cam_wrist": "observation.images.cam_wrist",
                        "observation.images.cam_side": "observation.images.cam_side",
                        "observation.state": "observation.state",
                        "action": "action",
                        "task": "task",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[franka_policy.FrankaInputs(model_type=model_config.model_type)],
            outputs=[franka_policy.FrankaOutputs()],
        )

        # Franka uses 8-dim joint space actions (7 joints + 1 gripper)
        # Apply delta transform to all 7 dimensions
        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),  # Franka数据集使用 "action" 而非 "actions"
        )
@dataclasses.dataclass(frozen=True)
class LeRobotFrankaEEDataConfig(DataConfigFactory):
    """Data config for Franka robot using EE (End-Effector) space control.

    EE space uses 7D actions: [x, y, z, roll, pitch, yaw, gripper]
    Unlike joint space, we use absolute actions (no delta transform) to avoid
    rotation interpolation issues.
    """
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Import EE policy
        from openpi.policies import franka_ee_policy

        # Repack transforms to standardize field names from LeRobot dataset
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.images.cam_high": "observation.images.cam_high",
                        "observation.images.cam_wrist": "observation.images.cam_wrist",
                        "observation.images.cam_side": "observation.images.cam_side",
                        "observation.state": "observation.state",
                        "action": "action",
                        "task_index": "task_index",  # Keep task_index for PromptFromLeRobotTask transform
                        "prompt": "prompt",  # Keep prompt added by PromptFromLeRobotTask
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[franka_ee_policy.FrankaEEInputs(model_type=model_config.model_type)],
            outputs=[franka_ee_policy.FrankaEEOutputs()],
        )

        # Franka EE uses 7-dim action space (3 xyz + 3 rpy + 1 gripper)
        # Use ABSOLUTE actions (no delta transform) to avoid rotation interpolation issues
        # delta_action_mask = _transforms.make_bool_mask(0, 0, 0, 0, 0, 0, 0)  # All False = all absolute
        # Note: We don't apply delta transform at all for EE space

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),  # Franka数据集使用 "action" 而非 "actions"
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDobotDataConfig(DataConfigFactory):
    """Data config for dual-arm Dobot robot.

    Dobot has 14 DOF (7 per arm) and 3 cameras (top, left_wrist, right_wrist).
    Uses delta actions for joints (not for grippers).
    """
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transforms to standardize field names from LeRobot dataset
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/top_image": "top_image",
                        "observation/left_wrist_image": "left_wrist_image",
                        "observation/right_wrist_image": "right_wrist_image",
                        "observation/state": "qpos",  # Dobot uses 'qpos' field
                        "actions": "actions",
                        "episode_index": "episode_index",
                        "frame_index": "frame_index",
                        "task_index": "task_index",  # Keep task_index for PromptFromLeRobotTask transform
                        "prompt": "prompt",  # Keep prompt added by PromptFromLeRobotTask
                    }
                )
            ]
        )

        # Data transforms for Dobot inputs/outputs
        data_transforms = _transforms.Group(
            inputs=[dobot_policy.DobotInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[dobot_policy.DobotOutputs()],
        )

        # Use delta actions for all 14 joints (last 2 are grippers, no delta)
        delta_action_mask = _transforms.make_bool_mask(14, -2)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            prompt_from_task=True,  # Use PromptFromLeRobotTask to map task_index to task text
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDobotFullDataConfig(DataConfigFactory):
    """Data config for dual-arm Dobot robot (full dataset format).

    Full datasets use different field naming:
    - observation.images.top/left_wrist/right_wrist instead of top_image/left_wrist_image/right_wrist_image
    - observation.state instead of qpos
    - action instead of actions
    """
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transforms to standardize field names from full dataset format
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/top_image": "observation.images.top",
                        "observation/left_wrist_image": "observation.images.left_wrist",
                        "observation/right_wrist_image": "observation.images.right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "episode_index": "episode_index",
                        "frame_index": "frame_index",
                        "task_index": "task_index",  # Keep task_index for PromptFromLeRobotTask transform
                        "prompt": "prompt",  # Keep prompt added by PromptFromLeRobotTask
                    }
                )
            ]
        )

        # Data transforms for Dobot inputs/outputs
        data_transforms = _transforms.Group(
            inputs=[dobot_policy.DobotInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[dobot_policy.DobotOutputs()],
        )

        # Use delta actions for all 14 joints (last 2 are grippers, no delta)
        delta_action_mask = _transforms.make_bool_mask(14, -2)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),  # Full dataset uses "action" instead of "actions"
            prompt_from_task=True,  # Use PromptFromLeRobotTask to map task_index to task text
        )
