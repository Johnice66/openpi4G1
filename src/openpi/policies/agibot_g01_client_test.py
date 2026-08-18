import numpy as np
import pytest

from examples.agibot_g01 import control_utils


# ==================== AgiBot G01 π0.5 adaptation: client safety tests BEGIN ====================
def test_parse_and_resample_action_chunk():
    actions = np.arange(32 * 16, dtype=np.float32).reshape(32, 16)
    parsed = control_utils.parse_action_chunk({"actions": actions})
    execution = control_utils.prepare_execution_chunk(parsed, execute_horizon=8, model_fps=30, control_hz=60)
    assert execution.shape == (16, 16)
    np.testing.assert_array_equal(execution[0], actions[0])
    np.testing.assert_array_equal(execution[-1], actions[7])


def test_invalid_action_is_rejected():
    with pytest.raises(ValueError, match="NaN"):
        control_utils.parse_action_chunk({"actions": np.full((32, 16), np.nan)})
    with pytest.raises(ValueError, match="shaped"):
        control_utils.parse_action_chunk({"actions": np.zeros((32, 15))})


def test_server_metadata_validation():
    metadata = {"robot_type": "agibot_g01", "action_horizon": 32, "policy_action_dim": 16, "dataset_fps": 30}
    control_utils.validate_server_metadata(metadata)
    metadata["action_horizon"] = 50
    with pytest.raises(ValueError, match="metadata mismatch"):
        control_utils.validate_server_metadata(metadata)


def test_server_metadata_validation_accepts_configurable_16_step_horizon():
    metadata = {"robot_type": "agibot_g01", "action_horizon": 16, "policy_action_dim": 16, "dataset_fps": 30}
    control_utils.validate_server_metadata(metadata, expected_horizon=16)
    with pytest.raises(ValueError, match="metadata mismatch"):
        control_utils.validate_server_metadata(metadata, expected_horizon=32)


def test_joint_limiter_restricts_acceleration_and_speed():
    limiter = control_utils.JointSafetyLimiter(
        dim=1, dt=0.1, max_step_delta=1.0, max_speed=2.0, max_accel=1.0
    )
    limiter.reset(np.array([0.0], dtype=np.float32))
    first = limiter(np.array([10.0], dtype=np.float32))
    second = limiter(np.array([10.0], dtype=np.float32))
    np.testing.assert_allclose(first, [0.01])
    np.testing.assert_allclose(second, [0.03])
# ==================== AgiBot G01 π0.5 adaptation: client safety tests END ====================
