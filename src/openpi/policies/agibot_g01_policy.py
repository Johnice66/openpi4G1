"""Input and output transforms for the AgiBot G01 bimanual robot."""

import dataclasses
from typing import Final

import einops
import numpy as np

from openpi import transforms


RAW_STATE_DIM: Final = 163
RAW_ACTION_DIM: Final = 36
POLICY_DIM: Final = 16
JOINT_ACTION_MASK: Final = transforms.make_bool_mask(14, -2)


def make_agibot_g01_example() -> dict:
    """Create an inference-format observation for smoke tests."""
    return {
        "images": {
            "top_head": np.zeros((800, 1280, 3), dtype=np.uint8),
            "hand_left": np.zeros((480, 848, 3), dtype=np.uint8),
            "hand_right": np.zeros((480, 848, 3), dtype=np.uint8),
        },
        "state": np.zeros(POLICY_DIM, dtype=np.float32),
        "prompt": "Fixed-point Non-generalized Door Opening",
    }


def _convert_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected an unbatched image with 3 dimensions, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = einops.rearrange(image, "c h w -> h w c")
    elif image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unable to determine image channel axis from shape {image.shape}")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.all(np.isfinite(image)):
            raise ValueError("Image contains NaN or Inf")
        # LeRobot v2.1 video decoding yields float32 in [0, 1].
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _select_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] == POLICY_DIM:
        selected = state
    elif state.shape[-1] == RAW_STATE_DIM:
        selected = np.concatenate((state[..., 28:42], state[..., 0:2]), axis=-1)
    else:
        raise ValueError(f"Expected state dimension {RAW_STATE_DIM} or {POLICY_DIM}, got {state.shape}")
    if not np.all(np.isfinite(selected)):
        raise ValueError("State contains NaN or Inf")
    return selected


def _select_actions(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[-1] == POLICY_DIM:
        selected = actions
    elif actions.shape[-1] == RAW_ACTION_DIM:
        selected = np.concatenate((actions[..., 16:30], actions[..., 0:2]), axis=-1)
    else:
        raise ValueError(f"Expected action dimension {RAW_ACTION_DIM} or {POLICY_DIM}, got {actions.shape}")
    if not np.all(np.isfinite(selected)):
        raise ValueError("Actions contain NaN or Inf")
    return selected


@dataclasses.dataclass(frozen=True)
class AgiBotG01Inputs(transforms.DataTransformFn):
    """Convert raw LeRobot or ROS-client data into the common openpi format."""

    def __call__(self, data: dict) -> dict:
        images = data.get("images")
        expected = {"top_head", "hand_left", "hand_right"}
        if not isinstance(images, dict) or set(images) != expected:
            got = set(images) if isinstance(images, dict) else type(images).__name__
            raise ValueError(f"Expected image keys {sorted(expected)}, got {got}")

        result = {
            "image": {
                "base_0_rgb": _convert_image(images["top_head"]),
                "left_wrist_0_rgb": _convert_image(images["hand_left"]),
                "right_wrist_0_rgb": _convert_image(images["hand_right"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": _select_state(data["state"]),
        }
        if "actions" in data:
            result["actions"] = _select_actions(data["actions"])
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class AgiBotG01Outputs(transforms.DataTransformFn):
    """Remove model padding and expose absolute G01 actions."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)[..., :POLICY_DIM]
        if actions.ndim != 2 or actions.shape[-1] != POLICY_DIM:
            raise ValueError(f"Expected action chunk shaped (T, {POLICY_DIM}), got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Model actions contain NaN or Inf")
        return {"actions": actions}
