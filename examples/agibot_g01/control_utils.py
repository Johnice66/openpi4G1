"""Pure NumPy validation and safety helpers for the AgiBot G01 client."""

from collections.abc import Mapping

import numpy as np


POLICY_ACTION_DIM = 16


def validate_server_metadata(metadata: Mapping, *, expected_horizon: int = 16, expected_fps: int = 30) -> None:
    expected = {
        "robot_type": "agibot_g01",
        "action_horizon": expected_horizon,
        "policy_action_dim": POLICY_ACTION_DIM,
        "dataset_fps": expected_fps,
    }
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches:
        raise ValueError(f"Policy server metadata mismatch: {mismatches}")


def parse_action_chunk(response: Mapping) -> np.ndarray:
    if "actions" not in response:
        raise KeyError(f"Policy response is missing 'actions'; keys={list(response)}")
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != POLICY_ACTION_DIM:
        raise ValueError(f"Expected actions shaped (T, {POLICY_ACTION_DIM}), got {actions.shape}")
    if actions.shape[0] == 0:
        raise ValueError("Policy returned an empty action chunk")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy actions contain NaN or Inf")
    return actions


def resample_sequence(sequence: np.ndarray, *, source_hz: float, target_hz: float) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[0] == 0:
        raise ValueError(f"Expected a non-empty (T, D) sequence, got {sequence.shape}")
    if source_hz <= 0 or target_hz <= 0:
        raise ValueError("source_hz and target_hz must be positive")
    if sequence.shape[0] == 1:
        return sequence.copy()
    output_steps = max(1, int(round(sequence.shape[0] * target_hz / source_hz)))
    if output_steps == sequence.shape[0]:
        return sequence.copy()
    src = np.arange(sequence.shape[0], dtype=np.float32)
    dst = np.linspace(0, sequence.shape[0] - 1, output_steps, dtype=np.float32)
    return np.stack([np.interp(dst, src, sequence[:, dim]) for dim in range(sequence.shape[1])], axis=-1).astype(
        np.float32
    )


def prepare_execution_chunk(
    actions: np.ndarray,
    *,
    execute_horizon: int,
    model_fps: float,
    control_hz: float,
) -> np.ndarray:
    actions = parse_action_chunk({"actions": actions})
    prefix = actions.shape[0] if execute_horizon <= 0 else min(execute_horizon, actions.shape[0])
    return resample_sequence(actions[:prefix], source_hz=model_fps, target_hz=control_hz)


class EMAFilter:
    def __init__(self, dim: int, alpha: float):
        if not 0 < alpha <= 1:
            raise ValueError("EMA alpha must be in (0, 1]")
        self._alpha = float(alpha)
        self._value = np.zeros(dim, dtype=np.float32)
        self._ready = False

    def reset(self, value: np.ndarray) -> None:
        self._value = np.asarray(value, dtype=np.float32).copy()
        self._ready = True

    def __call__(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if not self._ready:
            self.reset(value)
        else:
            self._value = self._alpha * value + (1.0 - self._alpha) * self._value
        return self._value.copy()


class JointSafetyLimiter:
    def __init__(self, *, dim: int, dt: float, max_step_delta: float, max_speed: float, max_accel: float):
        if min(dt, max_step_delta, max_speed, max_accel) <= 0:
            raise ValueError("Joint limiter parameters must be positive")
        self._dt = float(dt)
        self._max_delta = min(float(max_step_delta), float(max_speed) * self._dt)
        self._max_speed = float(max_speed)
        self._max_accel = float(max_accel)
        self._position = np.zeros(dim, dtype=np.float32)
        self._velocity = np.zeros(dim, dtype=np.float32)
        self._ready = False

    def reset(self, position: np.ndarray) -> None:
        self._position = np.asarray(position, dtype=np.float32).copy()
        self._velocity = np.zeros_like(self._position)
        self._ready = True

    def __call__(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32)
        if not np.all(np.isfinite(target)):
            raise ValueError("Joint target contains NaN or Inf")
        if not self._ready:
            self.reset(target)
            return self._position.copy()
        desired_delta = np.clip(target - self._position, -self._max_delta, self._max_delta)
        desired_velocity = np.clip(desired_delta / self._dt, -self._max_speed, self._max_speed)
        velocity_delta = np.clip(
            desired_velocity - self._velocity,
            -self._max_accel * self._dt,
            self._max_accel * self._dt,
        )
        velocity = np.clip(self._velocity + velocity_delta, -self._max_speed, self._max_speed)
        delta = np.clip(velocity * self._dt, -self._max_delta, self._max_delta)
        self._position = (self._position + delta).astype(np.float32)
        self._velocity = (delta / self._dt).astype(np.float32)
        return self._position.copy()


def blend_target(current: np.ndarray, target: np.ndarray, step: int, blend_steps: int) -> np.ndarray:
    if blend_steps <= 0:
        return np.asarray(target, dtype=np.float32)
    alpha = min(1.0, float(step + 1) / float(blend_steps))
    return (np.asarray(current) * (1.0 - alpha) + np.asarray(target) * alpha).astype(np.float32)
