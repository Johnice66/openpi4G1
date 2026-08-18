"""Run the AgiBot G01 training preflight and emit a non-destructive clean list.

The command scans every LeRobot episode before normalization or training. It
never edits parquet/video files. Confirmed bad episodes are written to a shared
``exclude_g01.txt`` file; warnings requiring human judgement are written to
``review_g01.txt``.

Example::

    uv run scripts/agibot_g01_data_quality.py \
      --dataset agibot/task_5867_203=dataset/task_5867_203 \
      --dataset agibot/task_5867_479=dataset/task_5867_479 \
      --video-check decode \
      --output-dir reports/g01_preflight

Use the generated exclusion file for both normalization and training::

    uv run scripts/agibot_g01_multi_train.py compute-norm ... \
      --exclude-file reports/g01_preflight/exclude_g01.txt

    uv run scripts/agibot_g01_multi_train.py train ... \
      --exclude-file reports/g01_preflight/exclude_g01.txt
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent import futures
import csv
import dataclasses
from datetime import UTC
from datetime import datetime
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import polars as pl
import tqdm

import openpi.training.episode_filter as _episode_filter

RAW_STATE_DIM = 163
RAW_ACTION_DIM = 36
JOINT_DIMS = 14
STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"
DEFAULT_FPS = 30.0


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    root: pathlib.Path


@dataclasses.dataclass(frozen=True)
class Thresholds:
    exclude_max_joint_delta: float = 0.35
    exclude_max_action_jump: float = 0.25
    exclude_max_state_jump: float = 0.25
    review_max_joint_delta: float = 0.25
    review_max_action_jump: float = 0.12
    review_max_state_jump: float = 0.12
    warning_ts_gap_ms: float = 50.0
    review_ts_gap_ms: float = 60.0
    moving_duplicate_threshold: float = 0.01


@dataclasses.dataclass
class EpisodeResult:
    dataset_id: str
    dataset_root: str
    episode_index: int
    parquet_path: str
    expected_frames: int | None
    rows: int = 0
    global_index_min: int | None = None
    global_index_max: int | None = None
    max_abs_joint_delta: float = math.nan
    max_abs_joint_delta_frame: int | None = None
    max_abs_joint_delta_joint: int | None = None
    max_action_jump: float = math.nan
    max_action_jump_frame: int | None = None
    max_action_jump_joint: int | None = None
    max_state_jump: float = math.nan
    max_state_jump_frame: int | None = None
    max_state_jump_joint: int | None = None
    max_ts_gap_ms: float = math.nan
    max_ts_gap_frame: int | None = None
    ts_gaps_over_warning: int = 0
    state_left_gripper_min: float = math.nan
    state_left_gripper_max: float = math.nan
    state_right_gripper_min: float = math.nan
    state_right_gripper_max: float = math.nan
    action_left_gripper_min: float = math.nan
    action_left_gripper_max: float = math.nan
    action_right_gripper_min: float = math.nan
    action_right_gripper_max: float = math.nan
    video_frames: dict[str, int | None] = dataclasses.field(default_factory=dict)
    duplicate_frame_candidates: dict[str, list[int]] = dataclasses.field(default_factory=dict)
    exclude_reasons: list[str] = dataclasses.field(default_factory=list)
    review_reasons: list[str] = dataclasses.field(default_factory=list)

    @property
    def status(self) -> str:
        if self.exclude_reasons:
            return "exclude"
        if self.review_reasons:
            return "review"
        return "pass"


@dataclasses.dataclass(frozen=True)
class VideoJob:
    dataset_id: str
    episode_index: int
    video_key: str
    path: pathlib.Path
    expected_frames: int


@dataclasses.dataclass(frozen=True)
class VideoResult:
    dataset_id: str
    episode_index: int
    video_key: str
    path: str
    decoded_frames: int | None
    duplicate_frame_candidates: list[int]
    error: str | None


def _parse_dataset(value: str) -> DatasetSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected --dataset DATASET_ID=LOCAL_ROOT")
    dataset_id, root = value.split("=", 1)
    dataset_id = dataset_id.strip()
    root = root.strip()
    if not dataset_id or not root:
        raise argparse.ArgumentTypeError("Both dataset id and local root are required")
    return DatasetSpec(dataset_id, pathlib.Path(root).expanduser().resolve())


def _series_to_2d_array(series: pl.Series, *, expected_dim: int, name: str) -> np.ndarray:
    values = series.to_numpy()
    if values.dtype == object:
        values = np.stack([np.asarray(item, dtype=np.float32) for item in series.to_list()])
    else:
        values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise ValueError(f"{name} expected (T, {expected_dim}), got {values.shape}")
    return values


def _scalar_array(series: pl.Series, *, dtype: np.dtype[Any]) -> np.ndarray:
    values = series.to_list()
    flattened = [item[0] if isinstance(item, (list, tuple)) else item for item in values]
    return np.asarray(flattened, dtype=dtype)


def _maximum_with_location(values: np.ndarray) -> tuple[float, int | None, int | None]:
    if values.size == 0:
        return 0.0, None, None
    flat_index = int(np.argmax(values))
    frame_index, joint_index = np.unravel_index(flat_index, values.shape)
    return float(values[frame_index, joint_index]), int(frame_index), int(joint_index)


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _load_episode_lengths(root: pathlib.Path) -> dict[int, int]:
    path = root / "meta" / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset is missing {path}")
    lengths: dict[int, int] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            episode_index = int(row["episode_index"])
            length = int(row["length"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid episode metadata at {path}:{line_number}") from error
        if episode_index in lengths:
            raise ValueError(f"Duplicate episode index {episode_index} in {path}")
        lengths[episode_index] = length
    return lengths


def _scan_episode(
    spec: DatasetSpec,
    path: pathlib.Path,
    expected_frames: int | None,
    *,
    fps: float,
    thresholds: Thresholds,
) -> tuple[EpisodeResult, np.ndarray]:
    episode_index = _episode_filter.episode_index_from_path(path)
    relative_path = path.relative_to(spec.root).as_posix()
    result = EpisodeResult(
        dataset_id=spec.dataset_id,
        dataset_root=str(spec.root),
        episode_index=episode_index,
        parquet_path=relative_path,
        expected_frames=expected_frames,
    )

    required_columns = [
        STATE_COLUMN,
        ACTION_COLUMN,
        "episode_index",
        "frame_index",
        "index",
        "ts",
        "timestamp",
    ]
    try:
        frame = pl.read_parquet(path, columns=required_columns)
        raw_state = _series_to_2d_array(frame[STATE_COLUMN], expected_dim=RAW_STATE_DIM, name=STATE_COLUMN)
        raw_action = _series_to_2d_array(frame[ACTION_COLUMN], expected_dim=RAW_ACTION_DIM, name=ACTION_COLUMN)
        result.rows = raw_state.shape[0]
        if raw_action.shape[0] != result.rows:
            raise ValueError(f"state/action row mismatch: {result.rows} vs {raw_action.shape[0]}")
        if result.rows == 0:
            raise ValueError("empty episode")

        episode_indices = _scalar_array(frame["episode_index"], dtype=np.int64)
        frame_indices = _scalar_array(frame["frame_index"], dtype=np.int64)
        global_indices = _scalar_array(frame["index"], dtype=np.int64)
        timestamps = _scalar_array(frame["timestamp"], dtype=np.float64)
        source_timestamps = _scalar_array(frame["ts"], dtype=np.uint64)

        if expected_frames is None or expected_frames != result.rows:
            _append_unique(result.exclude_reasons, f"length_mismatch(expected={expected_frames},actual={result.rows})")
        if not np.all(episode_indices == episode_index):
            _append_unique(result.exclude_reasons, "episode_index_mismatch")
        if not np.array_equal(frame_indices, np.arange(result.rows, dtype=np.int64)):
            _append_unique(result.exclude_reasons, "non_contiguous_frame_index")
        result.global_index_min = int(np.min(global_indices))
        result.global_index_max = int(np.max(global_indices))
        if not np.all(np.diff(global_indices) == 1):
            _append_unique(result.exclude_reasons, "non_contiguous_global_index_within_episode")
        expected_timestamps = np.arange(result.rows, dtype=np.float64) / fps
        if not np.allclose(timestamps, expected_timestamps, atol=1e-5, rtol=0):
            _append_unique(result.exclude_reasons, "timestamp_not_on_fps_grid")
        if not np.all(np.isfinite(raw_state)):
            _append_unique(result.exclude_reasons, "state_contains_nan_or_inf")
        if not np.all(np.isfinite(raw_action)):
            _append_unique(result.exclude_reasons, "action_contains_nan_or_inf")

        source_timestamp_diffs = np.diff(source_timestamps.astype(np.float64)) / 1e6
        if np.any(source_timestamp_diffs <= 0):
            _append_unique(result.exclude_reasons, "non_monotonic_source_timestamp")
        if source_timestamp_diffs.size:
            result.max_ts_gap_ms = float(np.max(source_timestamp_diffs))
            result.max_ts_gap_frame = int(np.argmax(source_timestamp_diffs)) + 1
            result.ts_gaps_over_warning = int(np.count_nonzero(source_timestamp_diffs > thresholds.warning_ts_gap_ms))
            if result.max_ts_gap_ms > thresholds.review_ts_gap_ms:
                _append_unique(
                    result.review_reasons,
                    f"source_timestamp_gap={result.max_ts_gap_ms:.3f}ms@{result.max_ts_gap_frame}",
                )

        state = raw_state[:, 28:42]
        action = raw_action[:, 16:30]
        joint_delta = np.abs(action - state)
        action_jump = np.abs(np.diff(action, axis=0))
        state_jump = np.abs(np.diff(state, axis=0))
        (
            result.max_abs_joint_delta,
            result.max_abs_joint_delta_frame,
            result.max_abs_joint_delta_joint,
        ) = _maximum_with_location(joint_delta)
        result.max_action_jump, action_jump_frame, result.max_action_jump_joint = _maximum_with_location(action_jump)
        result.max_state_jump, state_jump_frame, result.max_state_jump_joint = _maximum_with_location(state_jump)
        result.max_action_jump_frame = None if action_jump_frame is None else action_jump_frame + 1
        result.max_state_jump_frame = None if state_jump_frame is None else state_jump_frame + 1

        if result.max_abs_joint_delta > thresholds.exclude_max_joint_delta:
            _append_unique(result.exclude_reasons, f"max_joint_delta={result.max_abs_joint_delta:.6f}")
        elif result.max_abs_joint_delta > thresholds.review_max_joint_delta:
            _append_unique(result.review_reasons, f"max_joint_delta={result.max_abs_joint_delta:.6f}")
        if result.max_action_jump > thresholds.exclude_max_action_jump:
            _append_unique(result.exclude_reasons, f"max_action_jump={result.max_action_jump:.6f}")
        elif result.max_action_jump > thresholds.review_max_action_jump:
            _append_unique(result.review_reasons, f"max_action_jump={result.max_action_jump:.6f}")
        if result.max_state_jump > thresholds.exclude_max_state_jump:
            _append_unique(result.exclude_reasons, f"max_state_jump={result.max_state_jump:.6f}")
        elif result.max_state_jump > thresholds.review_max_state_jump:
            _append_unique(result.review_reasons, f"max_state_jump={result.max_state_jump:.6f}")

        result.state_left_gripper_min = float(np.min(raw_state[:, 0]))
        result.state_left_gripper_max = float(np.max(raw_state[:, 0]))
        result.state_right_gripper_min = float(np.min(raw_state[:, 1]))
        result.state_right_gripper_max = float(np.max(raw_state[:, 1]))
        result.action_left_gripper_min = float(np.min(raw_action[:, 0]))
        result.action_left_gripper_max = float(np.max(raw_action[:, 0]))
        result.action_right_gripper_min = float(np.min(raw_action[:, 1]))
        result.action_right_gripper_max = float(np.max(raw_action[:, 1]))

        state_step = np.pad(np.max(state_jump, axis=1), (1, 0))
        action_step = np.pad(np.max(action_jump, axis=1), (1, 0))
        motion_steps = np.maximum(state_step, action_step)
    except Exception as error:
        _append_unique(result.exclude_reasons, f"parquet_error={type(error).__name__}:{error}")
        motion_steps = np.zeros(result.rows, dtype=np.float32)

    return result, motion_steps


def _resolve_executable(explicit_path: pathlib.Path | None, name: str) -> str:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{name} executable does not exist: {path}")
        return str(path)
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(f"{name} is required for video checks; install FFmpeg or pass --{name}")
    return resolved


def _probe_video(job: VideoJob, *, executable: str, video_check: str) -> VideoResult:
    if not job.path.is_file():
        return VideoResult(
            job.dataset_id,
            job.episode_index,
            job.video_key,
            str(job.path),
            None,
            [],
            "missing_video",
        )

    if video_check in {"metadata", "decode"}:
        command = [executable, "-v", "error", "-threads", "1"]
        if video_check == "decode":
            command.append("-count_frames")
        field = "nb_read_frames" if video_check == "decode" else "nb_frames"
        command.extend(
            [
                "-select_streams",
                "v:0",
                "-show_entries",
                f"stream={field},avg_frame_rate",
                "-of",
                "json",
                str(job.path),
            ]
        )
        completed = subprocess.run(command, capture_output=True, check=False, text=True)
        try:
            stream = (json.loads(completed.stdout).get("streams") or [{}])[0]
            decoded_frames = int(stream[field])
            frame_rate = stream.get("avg_frame_rate")
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            decoded_frames = None
            frame_rate = None
        errors = []
        if completed.returncode:
            errors.append(f"exit={completed.returncode}")
        if completed.stderr.strip():
            errors.append(completed.stderr.strip())
        if frame_rate != "30/1":
            errors.append(f"fps={frame_rate}")
        return VideoResult(
            job.dataset_id,
            job.episode_index,
            job.video_key,
            str(job.path),
            decoded_frames,
            [],
            "; ".join(errors) or None,
        )

    command = [
        executable,
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        str(job.path),
        "-vf",
        "scale=64:64,format=gray",
        "-f",
        "framemd5",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, check=False, text=True)
    hashes = []
    for line in completed.stdout.splitlines():
        if line and not line.startswith("#"):
            parts = line.split(",")
            if len(parts) >= 6:
                hashes.append(parts[-1].strip())
    duplicates = [index for index in range(1, len(hashes)) if hashes[index] == hashes[index - 1]]
    errors = []
    if completed.returncode:
        errors.append(f"exit={completed.returncode}")
    if completed.stderr.strip():
        errors.append(completed.stderr.strip())
    return VideoResult(
        job.dataset_id,
        job.episode_index,
        job.video_key,
        str(job.path),
        len(hashes),
        duplicates,
        "; ".join(errors) or None,
    )


def _video_jobs(spec: DatasetSpec, info: dict[str, Any], lengths: dict[int, int]) -> list[VideoJob]:
    video_keys = [key for key, value in info.get("features", {}).items() if value.get("dtype") == "video"]
    if not video_keys:
        raise ValueError(f"No video features are declared in {spec.root / 'meta/info.json'}")
    jobs = []
    for episode_index, expected_frames in sorted(lengths.items()):
        episode_chunk = episode_index // int(info.get("chunks_size", 1000))
        for video_key in video_keys:
            path = spec.root / "videos" / f"chunk-{episode_chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"
            jobs.append(VideoJob(spec.dataset_id, episode_index, video_key, path, expected_frames))
    return jobs


def _apply_video_results(
    episode_results: dict[tuple[str, int], EpisodeResult],
    motion_steps: dict[tuple[str, int], np.ndarray],
    video_results: Sequence[VideoResult],
    *,
    thresholds: Thresholds,
) -> None:
    for video in video_results:
        result = episode_results[(video.dataset_id, video.episode_index)]
        result.video_frames[video.video_key] = video.decoded_frames
        if video.error:
            _append_unique(result.exclude_reasons, f"video_error[{video.video_key}]={video.error}")
        if video.decoded_frames != result.rows:
            _append_unique(
                result.exclude_reasons,
                f"video_frame_mismatch[{video.video_key}](video={video.decoded_frames},parquet={result.rows})",
            )
        if video.duplicate_frame_candidates:
            result.duplicate_frame_candidates[video.video_key] = video.duplicate_frame_candidates
            moving_duplicates = [
                index
                for index in video.duplicate_frame_candidates
                if index < result.rows - 1
                and index < motion_steps[(video.dataset_id, video.episode_index)].size
                and motion_steps[(video.dataset_id, video.episode_index)][index] > thresholds.moving_duplicate_threshold
            ]
            if moving_duplicates:
                _append_unique(
                    result.review_reasons,
                    f"moving_duplicate_frames[{video.video_key}]={moving_duplicates}",
                )


def _write_list(path: pathlib.Path, results: Sequence[EpisodeResult], *, review: bool) -> None:
    label = "review" if review else "exclude"
    lines = [
        f"# AgiBot G01 {label} list generated by scripts/agibot_g01_data_quality.py",
        "# dataset_id<TAB>episode_index<TAB>relative_parquet_path<TAB>reason",
    ]
    for result in sorted(results, key=lambda item: (item.dataset_id, item.episode_index)):
        reasons = result.review_reasons if review else result.exclude_reasons
        if not reasons:
            continue
        lines.append(
            "\t".join(
                [
                    result.dataset_id,
                    str(result.episode_index),
                    result.parquet_path,
                    ";".join(reasons),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _write_csv(path: pathlib.Path, results: Sequence[EpisodeResult]) -> None:
    rows = []
    for result in results:
        row = dataclasses.asdict(result)
        row["status"] = result.status
        row["video_frames"] = json.dumps(result.video_frames, sort_keys=True)
        row["duplicate_frame_candidates"] = json.dumps(result.duplicate_frame_candidates, sort_keys=True)
        row["exclude_reasons"] = ";".join(result.exclude_reasons)
        row["review_reasons"] = ";".join(result.review_reasons)
        rows.append(row)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: pathlib.Path,
    results: Sequence[EpisodeResult],
    *,
    generated_at: str,
    video_check: str,
    thresholds: Thresholds,
) -> None:
    dataset_ids = sorted({result.dataset_id for result in results})
    excluded = [result for result in results if result.exclude_reasons]
    review = [result for result in results if result.review_reasons and not result.exclude_reasons]
    lines = [
        "# AgiBot G01 训练前数据质量报告",
        "",
        f"- 生成时间: `{generated_at}`",
        f"- 视频检查: `{video_check}`",
        f"- 自动排除: {len(excluded)} 个 episode",
        f"- 人工复核: {len(review)} 个 episode",
        "",
        "## 数据集汇总",
        "",
        "| 数据集 | episode | 帧数 | 通过 | 复核 | 排除 | >50ms ts 间隔 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_id in dataset_ids:
        selected = [result for result in results if result.dataset_id == dataset_id]
        lines.append(
            f"| {dataset_id} | {len(selected)} | {sum(result.rows for result in selected)} "
            f"| {sum(result.status == 'pass' for result in selected)} "
            f"| {sum(result.status == 'review' for result in selected)} "
            f"| {sum(result.status == 'exclude' for result in selected)} "
            f"| {sum(result.ts_gaps_over_warning for result in selected)} |"
        )

    lines.extend(["", "## 自动排除", ""])
    if excluded:
        lines.extend(
            [
                "| 数据集 | Episode | joint delta | action jump | state jump | 原因 |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        lines.extend(
            (
                f"| {result.dataset_id} | {result.episode_index} | {result.max_abs_joint_delta:.6f} "
                f"| {result.max_action_jump:.6f} | {result.max_state_jump:.6f} "
                f"| {'; '.join(result.exclude_reasons)} |"
            )
            for result in excluded
        )
    else:
        lines.append("没有自动排除的 episode。")

    lines.extend(["", "## 人工复核", ""])
    if review:
        lines.extend(["| 数据集 | Episode | 原因 |", "|---|---:|---|"])
        lines.extend(
            f"| {result.dataset_id} | {result.episode_index} | {'; '.join(result.review_reasons)} |"
            for result in review
        )
    else:
        lines.append("没有需要人工复核的 episode。")

    lines.extend(
        [
            "",
            "## 自动排除阈值",
            "",
            "```json",
            json.dumps(dataclasses.asdict(thresholds), indent=2, ensure_ascii=False),
            "```",
            "",
            "`exclude_g01.txt` 必须同时传给归一化统计和训练命令。原始数据未被修改。",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", action="append", type=_parse_dataset, default=[], help="Repeat DATASET_ID=ROOT.")
    parser.add_argument(
        "--dataset-root",
        action="append",
        type=pathlib.Path,
        default=[],
        help="Single-dataset shortcut; id defaults to the directory name.",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--video-check", choices=("none", "metadata", "decode", "content"), default="decode")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--ffprobe", type=pathlib.Path, default=None)
    parser.add_argument("--ffmpeg", type=pathlib.Path, default=None)
    parser.add_argument("--exclude-review", action="store_true", help="Also put review episodes in exclude_g01.txt.")
    parser.add_argument("--fail-on-excluded", action="store_true")
    parser.add_argument("--max-exclude-fraction", type=float, default=0.03)
    parser.add_argument("--exclude-max-joint-delta", type=float, default=0.35)
    parser.add_argument("--exclude-max-action-jump", type=float, default=0.25)
    parser.add_argument("--exclude-max-state-jump", type=float, default=0.25)
    parser.add_argument("--review-max-joint-delta", type=float, default=0.25)
    parser.add_argument("--review-max-action-jump", type=float, default=0.12)
    parser.add_argument("--review-max-state-jump", type=float, default=0.12)
    parser.add_argument("--warning-ts-gap-ms", type=float, default=50.0)
    parser.add_argument("--review-ts-gap-ms", type=float, default=60.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    specs = list(args.dataset)
    specs.extend(
        DatasetSpec(root.expanduser().resolve().name, root.expanduser().resolve()) for root in args.dataset_root
    )
    if not specs:
        raise ValueError("At least one --dataset or --dataset-root is required")
    if len({spec.dataset_id for spec in specs}) != len(specs):
        raise ValueError("Dataset ids must be unique within one preflight run")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not 0 <= args.max_exclude_fraction <= 1:
        raise ValueError("--max-exclude-fraction must be between 0 and 1")

    thresholds = Thresholds(
        exclude_max_joint_delta=args.exclude_max_joint_delta,
        exclude_max_action_jump=args.exclude_max_action_jump,
        exclude_max_state_jump=args.exclude_max_state_jump,
        review_max_joint_delta=args.review_max_joint_delta,
        review_max_action_jump=args.review_max_action_jump,
        review_max_state_jump=args.review_max_state_jump,
        warning_ts_gap_ms=args.warning_ts_gap_ms,
        review_ts_gap_ms=args.review_ts_gap_ms,
    )
    generated_at = datetime.now(UTC).isoformat()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else pathlib.Path("reports/agibot_g01_data_quality") / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_results: dict[tuple[str, int], EpisodeResult] = {}
    motion_steps: dict[tuple[str, int], np.ndarray] = {}
    video_jobs = []
    dataset_summaries = []

    for spec in specs:
        info_path = spec.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Dataset is missing {info_path}")
        info = json.loads(info_path.read_text())
        fps = float(info.get("fps", DEFAULT_FPS))
        if fps <= 0:
            raise ValueError(f"Invalid dataset FPS in {info_path}: {fps}")
        features = info.get("features", {})
        if features.get(STATE_COLUMN, {}).get("shape") != [RAW_STATE_DIM]:
            raise ValueError(f"{info_path} does not declare {STATE_COLUMN} shape [{RAW_STATE_DIM}]")
        if features.get(ACTION_COLUMN, {}).get("shape") != [RAW_ACTION_DIM]:
            raise ValueError(f"{info_path} does not declare {ACTION_COLUMN} shape [{RAW_ACTION_DIM}]")
        lengths = _load_episode_lengths(spec.root)
        parquet_files = sorted((spec.root / "data").glob("chunk-*/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {spec.root / 'data'}")

        declared_ids = set(lengths)
        actual_ids = {_episode_filter.episode_index_from_path(path) for path in parquet_files}
        if declared_ids != actual_ids:
            raise ValueError(
                f"Parquet/metadata episode mismatch for {spec.dataset_id}: "
                f"missing={sorted(declared_ids - actual_ids)}, extra={sorted(actual_ids - declared_ids)}"
            )

        for path in tqdm.tqdm(parquet_files, desc=f"Parquet preflight: {spec.dataset_id}"):
            episode_index = _episode_filter.episode_index_from_path(path)
            result, episode_motion = _scan_episode(
                spec,
                path,
                lengths.get(episode_index),
                fps=fps,
                thresholds=thresholds,
            )
            episode_results[(spec.dataset_id, episode_index)] = result
            motion_steps[(spec.dataset_id, episode_index)] = episode_motion

        selected_results = sorted(
            (result for key, result in episode_results.items() if key[0] == spec.dataset_id),
            key=lambda result: result.episode_index,
        )
        expected_global_index = 0
        for result in selected_results:
            if result.global_index_min != expected_global_index:
                _append_unique(
                    result.exclude_reasons,
                    f"global_index_gap(expected={expected_global_index},actual={result.global_index_min})",
                )
            if result.global_index_max is not None:
                expected_global_index = result.global_index_max + 1

        actual_frames = sum(result.rows for result in selected_results)
        if info.get("total_episodes") != len(parquet_files):
            raise ValueError(
                f"Metadata episode total mismatch for {spec.dataset_id}: "
                f"declared={info.get('total_episodes')}, actual={len(parquet_files)}"
            )
        if info.get("total_frames") != actual_frames:
            raise ValueError(
                f"Metadata frame total mismatch for {spec.dataset_id}: "
                f"declared={info.get('total_frames')}, actual={actual_frames}"
            )

        if args.video_check != "none":
            video_jobs.extend(_video_jobs(spec, info, lengths))
        dataset_summaries.append(
            {
                "dataset_id": spec.dataset_id,
                "root": str(spec.root),
                "declared_episodes": info.get("total_episodes"),
                "declared_frames": info.get("total_frames"),
                "actual_episodes": len(parquet_files),
                "actual_frames": actual_frames,
            }
        )

    if video_jobs:
        executable = (
            _resolve_executable(args.ffmpeg, "ffmpeg")
            if args.video_check == "content"
            else _resolve_executable(args.ffprobe, "ffprobe")
        )
        video_results = []
        with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending = [
                executor.submit(_probe_video, job, executable=executable, video_check=args.video_check)
                for job in video_jobs
            ]
            video_results.extend(
                future.result()
                for future in tqdm.tqdm(
                    futures.as_completed(pending),
                    total=len(pending),
                    desc=f"Video preflight ({args.video_check})",
                )
            )
        _apply_video_results(episode_results, motion_steps, video_results, thresholds=thresholds)

    results = sorted(episode_results.values(), key=lambda result: (result.dataset_id, result.episode_index))
    excluded = [result for result in results if result.exclude_reasons]
    reviews = [result for result in results if result.review_reasons and not result.exclude_reasons]
    exclusion_output = list(excluded)
    if args.exclude_review:
        for result in reviews:
            result.exclude_reasons.extend(result.review_reasons)
        exclusion_output.extend(reviews)

    _write_list(output_dir / "exclude_g01.txt", exclusion_output, review=False)
    _write_list(output_dir / "review_g01.txt", reviews, review=True)
    _write_csv(output_dir / "episode_metrics.csv", results)
    (output_dir / "thresholds.json").write_text(json.dumps(dataclasses.asdict(thresholds), indent=2) + "\n")
    report = {
        "generated_at": generated_at,
        "video_check": args.video_check,
        "thresholds": dataclasses.asdict(thresholds),
        "datasets": dataset_summaries,
        "summary": {
            "episodes": len(results),
            "frames": sum(result.rows for result in results),
            "passed": sum(result.status == "pass" for result in results),
            "review": sum(result.status == "review" for result in results),
            "excluded": sum(result.status == "exclude" for result in results),
        },
        "episodes": [dict(dataclasses.asdict(result), status=result.status) for result in results],
    }
    (output_dir / "data_quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(
        output_dir / "data_quality_report.md",
        results,
        generated_at=generated_at,
        video_check=args.video_check,
        thresholds=thresholds,
    )

    exclude_fraction = len(exclusion_output) / len(results)
    print(f"Report directory: {output_dir}")
    print(
        f"Episodes: {len(results)}, pass={sum(result.status == 'pass' for result in results)}, "
        f"review={len(reviews)}, exclude={len(exclusion_output)}"
    )
    print(f"Use exclusion file: {output_dir / 'exclude_g01.txt'}")
    if exclude_fraction > args.max_exclude_fraction:
        print(
            f"Refusing silent continuation: excluded fraction {exclude_fraction:.2%} exceeds "
            f"--max-exclude-fraction {args.max_exclude_fraction:.2%}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.fail_on_excluded and exclusion_output:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
