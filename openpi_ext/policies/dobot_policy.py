import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # Only rearrange if image is in (C, H, W) format
    # Video format datasets already provide (H, W, C) format
    if len(image.shape) == 3 and image.shape[0] == 3 and image.shape[2] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DobotInputs(transforms.DataTransformFn):
    """Transform for dual-arm Dobot robot inputs.

    Dobot has 14 DOF total (7 per arm) + 3 cameras (top, left_wrist, right_wrist).
    """
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Get the 14D state (7 per arm)
        ee_state = data["observation/state"]
        state = transforms.pad_to_dim(ee_state, self.action_dim)

        # Parse images to uint8 (H,W,C) format
        top_image = _parse_image(data["observation/top_image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": top_image,  # Use 'base_0_rgb' to match model expectations
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]
        if "frame_index" in data:
            inputs["frame_index"] = data["frame_index"]

        # Actions are only available during training
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (language instruction) to the model
        # Only set prompt if it exists in data (don't set default during norm stats computation)
        if "task" in data:
            inputs["prompt"] = data["task"]
        elif "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DobotOutputs(transforms.DataTransformFn):
    """Transform for dual-arm Dobot robot outputs.

    Returns first 14 dimensions (7 per arm).
    """
    def __call__(self, data: dict) -> dict:
        # Return only the first 14 dims (dual-arm actions)
        return {"actions": np.asarray(data["actions"][:, :14])}
