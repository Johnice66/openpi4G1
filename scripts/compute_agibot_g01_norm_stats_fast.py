"""Compute AgiBot G01 normalization stats without decoding videos.

This is a fast path for the ``pi05_agibot_g01`` data contract. The regular
``scripts/compute_norm_stats.py`` goes through the full LeRobot/openpi transform
pipeline, which also materializes image observations before only accumulating
``state`` and ``actions``. This script reads the local LeRobot parquet files
directly and selects only:

- ``observation.state``
- ``action``

It then applies the same G01 state/action slicing and joint delta conversion used
by ``LeRobotAgiBotG01DataConfig``:

- state: ``state[28:42] + state[0:2]``
- action: ``action[16:30] + action[0:2]``
- action chunk horizon: 32 by default
- joint delta: first 14 dimensions
- gripper absolute: last 2 dimensions

Output defaults to an asset id inferred from the local dataset directory name.
For example, ``--dataset-root ./dataset/task_6030`` writes:
``assets/pi05_agibot_g01/agibot/task_6030/norm_stats.json``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import pathlib

import numpy as np
import polars as pl
import tqdm

from openpi.policies import agibot_g01_policy
import openpi.shared.normalize as normalize
import openpi.training.episode_filter as _episode_filter

RAW_STATE_DIM = 163
RAW_ACTION_DIM = 36
POLICY_DIM = 16
JOINT_DIMS = 14
DEFAULT_ACTION_HORIZON = 32
STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"
DEFAULT_CONFIG_NAME = "pi05_agibot_g01"
DEFAULT_ASSET_NAMESPACE = "agibot"


# ==================== AgiBot G01 π0.5 adaptation: explicit asset id management BEGIN ====================
def _infer_asset_id(dataset_root: pathlib.Path) -> str:
    """Infer the default asset id from a local LeRobot task directory."""

    task_name = dataset_root.name
    if not task_name:
        raise ValueError(f"Cannot infer asset id from dataset root: {dataset_root}")
    return f"{DEFAULT_ASSET_NAMESPACE}/{task_name}"


def _default_output_dir(config_name: str, asset_id: str) -> pathlib.Path:
    return (pathlib.Path("assets") / config_name / asset_id).resolve()
# ==================== AgiBot G01 π0.5 adaptation: explicit asset id management END ====================


def _iter_parquet_files(dataset_root: pathlib.Path) -> list[pathlib.Path]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Local LeRobot dataset root does not contain meta/info.json: {dataset_root}")

    files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    return files


def _series_to_2d_array(series: pl.Series, *, expected_dim: int, column_name: str) -> np.ndarray:
    values = series.to_numpy()
    if values.dtype == object:
        values = np.stack([np.asarray(item, dtype=np.float32) for item in series.to_list()], axis=0)
    else:
        values = np.asarray(values, dtype=np.float32)

    if values.ndim != 2 or values.shape[-1] != expected_dim:
        raise ValueError(f"Expected {column_name!r} shaped (T, {expected_dim}), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{column_name!r} contains NaN or Inf")
    return values


def _select_state(raw_state: np.ndarray) -> np.ndarray:
    if raw_state.shape[-1] != RAW_STATE_DIM:
        raise ValueError(f"Expected raw state dim {RAW_STATE_DIM}, got {raw_state.shape}")
    return np.concatenate((raw_state[..., 28:42], raw_state[..., 0:2]), axis=-1).astype(np.float32, copy=False)


def _select_action(raw_action: np.ndarray) -> np.ndarray:
    if raw_action.shape[-1] != RAW_ACTION_DIM:
        raise ValueError(f"Expected raw action dim {RAW_ACTION_DIM}, got {raw_action.shape}")
    return np.concatenate((raw_action[..., 16:30], raw_action[..., 0:2]), axis=-1).astype(np.float32, copy=False)


def _make_action_chunks(actions: np.ndarray, horizon: int) -> np.ndarray:
    if actions.ndim != 2 or actions.shape[-1] != POLICY_DIM:
        raise ValueError(f"Expected selected actions shaped (T, {POLICY_DIM}), got {actions.shape}")
    if horizon <= 0:
        raise ValueError("--action-horizon must be positive")
    if actions.shape[0] == 0:
        raise ValueError("Cannot build action chunks for an empty episode")

    padding = np.repeat(actions[-1:], horizon - 1, axis=0)
    padded = np.concatenate((actions, padding), axis=0)
    return np.stack([padded[offset : offset + actions.shape[0]] for offset in range(horizon)], axis=1)


def _apply_joint_delta(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    if actions.shape[:1] != state.shape[:1]:
        raise ValueError(f"Action/state row count mismatch: actions={actions.shape}, state={state.shape}")
    actions = actions.copy()
    actions[..., :JOINT_DIMS] -= state[:, None, :JOINT_DIMS]
    return actions


def _read_episode_arrays(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pl.read_parquet(path, columns=[STATE_COLUMN, ACTION_COLUMN])
    raw_state = _series_to_2d_array(frame[STATE_COLUMN], expected_dim=RAW_STATE_DIM, column_name=STATE_COLUMN)
    raw_action = _series_to_2d_array(frame[ACTION_COLUMN], expected_dim=RAW_ACTION_DIM, column_name=ACTION_COLUMN)
    if raw_state.shape[0] != raw_action.shape[0]:
        raise ValueError(f"State/action row count mismatch in {path}: {raw_state.shape[0]} vs {raw_action.shape[0]}")
    return raw_state, raw_action


def _update_stats_from_files(
    files: Iterable[pathlib.Path],
    *,
    action_horizon: int,
    max_frames: int | None,
) -> tuple[dict[str, normalize.RunningStats], int]:
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    processed = 0

    for path in tqdm.tqdm(list(files), desc="Computing G01 fast stats"):
        raw_state, raw_action = _read_episode_arrays(path)
        state = _select_state(raw_state)
        actions = _select_action(raw_action)
        action_chunks = _apply_joint_delta(_make_action_chunks(actions, action_horizon), state)

        if max_frames is not None:
            remaining = max_frames - processed
            if remaining <= 0:
                break
            state = state[:remaining]
            action_chunks = action_chunks[:remaining]

        stats["state"].update(state)
        stats["actions"].update(action_chunks)
        processed += state.shape[0]

    if processed < 2:
        raise ValueError(f"Need at least 2 frames to compute stats, got {processed}")
    return stats, processed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True, help="Local LeRobot task root.")
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Used only to build the default output path.",
    )
    # ==================== AgiBot G01 π0.5 adaptation: asset id CLI BEGIN ====================
    parser.add_argument(
        "--asset-id",
        "--repo-id",
        dest="asset_id",
        default=None,
        help=(
            "Asset id used only to build the default output path. Defaults to agibot/<dataset-root-name>. "
            "--repo-id is kept as a backward-compatible alias."
        ),
    )
    # ==================== AgiBot G01 π0.5 adaptation: asset id CLI END ====================
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Directory that will receive norm_stats.json.",
    )
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional deterministic prefix subset for quick checks.",
    )
    parser.add_argument(
        "--exclude-file",
        type=pathlib.Path,
        default=None,
        help="Episode list emitted by scripts/agibot_g01_data_quality.py.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when set")

    dataset_root = args.dataset_root.expanduser().resolve()
    asset_id = args.asset_id or _infer_asset_id(dataset_root)
    files = _iter_parquet_files(dataset_root)
    files, exclusions = _episode_filter.filter_parquet_files(
        files,
        repo_id=asset_id,
        dataset_root=dataset_root,
        exclusion_file=args.exclude_file,
    )
    if exclusions:
        print(f"Excluded episodes: {sorted(exclusions)}")
    stats, processed = _update_stats_from_files(
        files,
        action_horizon=args.action_horizon,
        max_frames=args.max_frames,
    )
    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    # ==================== AgiBot G01 π0.5 adaptation: physical gripper normalization BEGIN ====================
    norm_stats = agibot_g01_policy.apply_gripper_physical_norm_ranges(norm_stats)
    # ==================== AgiBot G01 π0.5 adaptation: physical gripper normalization END ====================

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else _default_output_dir(args.config_name, asset_id)
    )
    print(f"Processed frames: {processed}")
    print(f"Asset id: {asset_id}")
    print(f"Writing stats to: {output_dir}")
    normalize.save(output_dir, norm_stats)


if __name__ == "__main__":
    main()
