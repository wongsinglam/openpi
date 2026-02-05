import dataclasses
from pathlib import Path

import einops
import numpy as np
import torch

from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the Franka policy."""
    return {
        "observation.state": np.random.rand(8),  # 8维: 7 joints + 1 gripper
        "observation.images.cam_high": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.cam_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.cam_side": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "task": "grasp the object and place it in the basket",
    }


def _decode_video_frame(video_path: str, frame_index: int = 0) -> np.ndarray:
    """Decode a single frame from a video file."""
    import av
    import os

    original_path = video_path
    # Handle relative paths - prepend dataset root
    if not Path(video_path).is_absolute():
        # Check if HF_LEROBOT_HOME is set
        lerobot_home = Path(os.environ.get("HF_LEROBOT_HOME", str(Path.home() / ".cache" / "huggingface" / "lerobot")))

        # Try to find the video file in any Franka dataset
        # List all directories that start with 'franka'
        possible_datasets = [d.name for d in lerobot_home.iterdir() if d.is_dir() and d.name.startswith('franka')]

        # Try multiple path variations
        path_variations = [
            video_path,  # Original path
            # If path is videos/cam/episode.mp4, try videos/chunk-000/cam/episode.mp4
            video_path.replace('videos/', 'videos/chunk-000/', 1) if video_path.startswith('videos/') else None,
        ]
        path_variations = [p for p in path_variations if p]  # Remove None

        for dataset_name in possible_datasets:
            for path_var in path_variations:
                candidate_path = lerobot_home / dataset_name / path_var
                if candidate_path.exists():
                    video_path = str(candidate_path)
                    break
            if Path(video_path).is_absolute():  # Found it
                break
        else:
            # If not found, try with the relative path as-is (might be wrong but will give better error message)
            error_msg = f"Could not find video file.\nOriginal path: {original_path}\nSearched in datasets: {possible_datasets}\nPath variations tried: {path_variations}\nlerobot_home: {lerobot_home}"
            raise FileNotFoundError(error_msg)

    container = av.open(video_path)
    video_stream = container.streams.video[0]

    # Seek to the desired frame
    for i, frame in enumerate(container.decode(video_stream)):
        if i == frame_index:
            # Convert to numpy array in RGB format
            img = frame.to_ndarray(format='rgb24')
            container.close()
            return img

    container.close()
    raise ValueError(f"Could not find frame {frame_index} in video {video_path}")


def _parse_image(image) -> np.ndarray:
    # Handle video paths (strings)
    if isinstance(image, str):
        # This is a video path, decode the first frame
        return _decode_video_frame(image, frame_index=0)

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation.images.cam_high" or "observation.images.cam_wrist",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation.images.cam_high"])
        wrist_image = _parse_image(data["observation.images.cam_wrist"])
        side_image = _parse_image(data["observation.images.cam_side"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation.state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Franka has a side camera, so we use it for the right wrist view.
                # If you don't want to use the side camera, you can replace it with zeros:
                # "right_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": side_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "action" in data:
            inputs["actions"] = data["action"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        # Franka dataset uses "task" field instead of "prompt".
        if "task" in data:
            inputs["prompt"] = data["task"]
        elif "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Franka, we return 8 actions (7 joint positions + 1 gripper).
        # Joint space control: [joint1, joint2, joint3, joint4, joint5, joint6, joint7, gripper]
        return {"actions": np.asarray(data["actions"][:, :8])}
