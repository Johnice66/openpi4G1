import dataclasses
import os

import numpy as np
import pytest

from openpi import transforms
from openpi.policies import agibot_g01_policy
from openpi.shared import normalize
from openpi.training import config as training_config
from openpi.training import data_loader


# ==================== AgiBot G01 π0.5 adaptation: transform unit tests BEGIN ====================
def test_raw_dataset_slices_and_images():
    state = np.arange(agibot_g01_policy.RAW_STATE_DIM, dtype=np.float32)
    actions = np.arange(2 * agibot_g01_policy.RAW_ACTION_DIM, dtype=np.float32).reshape(2, -1)
    data = {
        "images": {
            "top_head": np.zeros((3, 10, 20), dtype=np.float32),
            "hand_left": np.zeros((3, 8, 12), dtype=np.float32),
            "hand_right": np.zeros((3, 8, 12), dtype=np.float32),
        },
        "state": state,
        "actions": actions,
        "prompt": np.asarray("open the door"),
    }

    result = agibot_g01_policy.AgiBotG01Inputs()(data)

    np.testing.assert_array_equal(result["state"], np.concatenate((state[28:42], state[0:2])))
    np.testing.assert_array_equal(
        result["actions"], np.concatenate((actions[:, 16:30], actions[:, 0:2]), axis=-1)
    )
    assert result["image"]["base_0_rgb"].shape == (10, 20, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8


def test_inference_format_and_delta_round_trip():
    result = agibot_g01_policy.AgiBotG01Inputs()(agibot_g01_policy.make_agibot_g01_example())
    absolute = np.arange(3 * agibot_g01_policy.POLICY_DIM, dtype=np.float32).reshape(3, -1)
    item = {"state": result["state"].copy(), "actions": absolute.copy()}

    transforms.DeltaActions(agibot_g01_policy.JOINT_ACTION_MASK)(item)
    np.testing.assert_array_equal(item["actions"][:, 14:], absolute[:, 14:])
    transforms.AbsoluteActions(agibot_g01_policy.JOINT_ACTION_MASK)(item)

    np.testing.assert_allclose(item["actions"], absolute)
    assert set(result["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}


def test_outputs_remove_model_padding():
    result = agibot_g01_policy.AgiBotG01Outputs()({"actions": np.zeros((16, 32), dtype=np.float32)})
    assert result["actions"].shape == (16, 16)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.zeros(15, dtype=np.float32), "Expected state dimension"),
        (np.full(16, np.nan, dtype=np.float32), "State contains"),
    ],
)
def test_invalid_state_rejected(value, message):
    example = agibot_g01_policy.make_agibot_g01_example()
    example["state"] = value
    with pytest.raises(ValueError, match=message):
        agibot_g01_policy.AgiBotG01Inputs()(example)


def test_training_config_uses_requested_long_horizon_defaults():
    config = training_config.get_config("pi05_agibot_g01")

    assert config.model.action_horizon == 32
    assert config.batch_size == 64
    assert config.num_train_steps == 100_000
    assert config.fsdp_devices == 2
    assert config.lr_schedule.decay_steps == 100_000
    assert config.policy_metadata["action_horizon"] == 32


# ==================== AgiBot G01 π0.5 adaptation: independent 16-step config test BEGIN ====================
def test_h16_training_config_is_independent_and_consistent():
    config = training_config.get_config("pi05_agibot_g01_h16")

    assert config.name == "pi05_agibot_g01_h16"
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 16
    assert config.batch_size == 64
    assert config.num_train_steps == 100_000
    assert config.fsdp_devices == 2
    assert config.lr_schedule.decay_steps == 100_000
    assert config.policy_metadata["action_horizon"] == 16
    assert config.policy_metadata["policy_action_dim"] == 16


# ==================== AgiBot G01 π0.5 adaptation: independent 16-step config test END ====================


def test_gripper_norm_stats_use_confirmed_physical_ranges():
    collapsed = {
        "state": normalize.NormStats(
            mean=np.zeros(16),
            std=np.ones(16),
            q01=np.zeros(16),
            q99=np.ones(16),
        ),
        "actions": normalize.NormStats(
            mean=np.zeros(16),
            std=np.ones(16),
            q01=np.zeros(16),
            q99=np.ones(16),
        ),
    }
    collapsed["state"].q01[14:16] = 119.99788
    collapsed["state"].q99[14:16] = 119.99788
    collapsed["actions"].q01[14:16] = 0.999934
    collapsed["actions"].q99[14:16] = 0.999934

    fixed = agibot_g01_policy.apply_gripper_physical_norm_ranges(collapsed)

    np.testing.assert_array_equal(fixed["state"].q01[14:16], [0.0, 0.0])
    np.testing.assert_array_equal(fixed["state"].q99[14:16], [120.0, 120.0])
    np.testing.assert_array_equal(fixed["actions"].q01[14:16], [0.0, 0.0])
    np.testing.assert_array_equal(fixed["actions"].q99[14:16], [1.0, 1.0])
    np.testing.assert_array_equal(collapsed["state"].q01[14:16], [119.99788, 119.99788])
    np.testing.assert_array_equal(collapsed["actions"].q01[14:16], [0.999934, 0.999934])

    sample = {
        "state": np.concatenate([np.zeros(14), [0.0, 120.0]]),
        "actions": np.concatenate([np.zeros((2, 14)), [[0.0, 1.0], [1.0, 0.0]]], axis=-1),
    }
    normalized = transforms.Normalize(fixed, use_quantiles=True)(sample)
    np.testing.assert_allclose(normalized["state"][14:16], [-1.0, 1.0])
    np.testing.assert_allclose(normalized["actions"][:, 14:16], [[-1.0, 1.0], [1.0, -1.0]])

    restored = transforms.Unnormalize(fixed, use_quantiles=True)(normalized)
    np.testing.assert_allclose(restored["state"][14:16], [0.0, 120.0], atol=1e-4)
    np.testing.assert_allclose(restored["actions"][:, 14:16], [[0.0, 1.0], [1.0, 0.0]], atol=1e-5)
# ==================== AgiBot G01 π0.5 adaptation: transform unit tests END ====================


# ==================== AgiBot G01 π0.5 adaptation: real dataset smoke test BEGIN ====================
@pytest.mark.manual
def test_real_task_5093_sample():
    dataset_root = os.environ.get("OPENPI_G01_TEST_DATASET_ROOT")
    if not dataset_root:
        pytest.skip("Set OPENPI_G01_TEST_DATASET_ROOT to run the real-dataset smoke test")
    config = training_config.get_config("pi05_agibot_g01")
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, dataset_root=dataset_root),
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transformed = data_loader.TransformedDataset(
        dataset,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs],
    )

    sample = transformed[0]

    assert sample["state"].shape == (16,)
    assert sample["actions"].shape == (32, 16)
    assert set(sample["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert str(sample["prompt"]) == "Fixed-point Non-generalized Door Opening"
# ==================== AgiBot G01 π0.5 adaptation: real dataset smoke test END ====================
