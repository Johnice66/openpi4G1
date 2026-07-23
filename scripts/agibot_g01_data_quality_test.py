# ruff: noqa: SLF001

import pathlib

import agibot_g01_data_quality as quality
import numpy as np
import polars as pl


def _write_episode(
    path: pathlib.Path,
    *,
    action_jump: float,
    source_timestamp_step: int = 33_333_333,
    rows: int = 8,
) -> None:
    state = np.zeros((rows, quality.RAW_STATE_DIM), dtype=np.float32)
    action = np.zeros((rows, quality.RAW_ACTION_DIM), dtype=np.float32)
    action[3, 16 + 12] = action_jump
    frame = pl.DataFrame(
        {
            quality.STATE_COLUMN: state.tolist(),
            quality.ACTION_COLUMN: action.tolist(),
            "episode_index": [0] * rows,
            "frame_index": list(range(rows)),
            "index": list(range(rows)),
            "ts": [[1_000_000_000 + index * source_timestamp_step] for index in range(rows)],
            "timestamp": [index / 30 for index in range(rows)],
        }
    )
    path.parent.mkdir(parents=True)
    frame.write_parquet(path)


def test_control_jump_is_automatically_excluded(tmp_path):
    root = tmp_path / "task_5867_479"
    path = root / "data/chunk-000/episode_000000.parquet"
    _write_episode(path, action_jump=0.5)

    result, _ = quality._scan_episode(
        quality.DatasetSpec("agibot/task_5867_479", root),
        path,
        expected_frames=8,
        fps=30,
        thresholds=quality.Thresholds(),
    )

    assert result.status == "exclude"
    assert result.max_action_jump == 0.5
    assert "max_action_jump=0.500000" in result.exclude_reasons
    assert "max_joint_delta=0.500000" in result.exclude_reasons


def test_normal_episode_passes(tmp_path):
    root = tmp_path / "task_5867"
    path = root / "data/chunk-000/episode_000000.parquet"
    _write_episode(path, action_jump=0.01)

    result, _ = quality._scan_episode(
        quality.DatasetSpec("agibot/task_5867", root),
        path,
        expected_frames=8,
        fps=30,
        thresholds=quality.Thresholds(),
    )

    assert result.status == "pass"


def test_source_timestamp_gap_requires_review(tmp_path):
    root = tmp_path / "task_5867"
    path = root / "data/chunk-000/episode_000000.parquet"
    _write_episode(path, action_jump=0.01, source_timestamp_step=70_000_000)

    result, _ = quality._scan_episode(
        quality.DatasetSpec("agibot/task_5867", root),
        path,
        expected_frames=8,
        fps=30,
        thresholds=quality.Thresholds(),
    )

    assert result.status == "review"
    assert result.ts_gaps_over_warning == 7
    assert result.review_reasons == ["source_timestamp_gap=70.000ms@1"]


def test_episode_length_mismatch_is_excluded(tmp_path):
    root = tmp_path / "task_5867"
    path = root / "data/chunk-000/episode_000000.parquet"
    _write_episode(path, action_jump=0.01)

    result, _ = quality._scan_episode(
        quality.DatasetSpec("agibot/task_5867", root),
        path,
        expected_frames=9,
        fps=30,
        thresholds=quality.Thresholds(),
    )

    assert result.status == "exclude"
    assert "length_mismatch(expected=9,actual=8)" in result.exclude_reasons
