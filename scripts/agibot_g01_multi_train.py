"""AgiBot G01 multi-LeRobot training helper.

This script keeps the original LeRobot datasets unchanged and mixes them only at
training time. It is intentionally self-contained so the standard openpi training
code path remains untouched.

Typical workflow:

1. Compute one normalization asset over the mixed datasets:

   uv run scripts/agibot_g01_multi_train.py compute-norm \
     --dataset agibot/task_5093=/path/to/task_5093 \
     --dataset agibot/task_6030=/path/to/task_6030 \
     --asset-id agibot/g01_mix_5093_6030

2. Train with the same mixed asset:

   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/agibot_g01_multi_train.py train \
     --dataset agibot/task_5093=/path/to/task_5093 \
     --dataset agibot/task_6030=/path/to/task_6030 \
     --asset-id agibot/g01_mix_5093_6030 \
     --exp-name g01_mix_5093_6030 \
     --overwrite

By default, training samples frames proportional to each dataset size. Add
``--equal-dataset-sampling`` or explicit repeated ``--sampling-weight`` values
when each task should contribute equally regardless of frame count.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import importlib.util
import logging
import pathlib
import types

import jax
import numpy as np
import torch
import tqdm

import compute_agibot_g01_norm_stats_fast as _fast_stats
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


DEFAULT_CONFIG_NAME = "pi05_agibot_g01"


# ==================== AgiBot G01 multi-dataset helper BEGIN ====================
@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    """One local LeRobot dataset participating in the mixed training run."""

    repo_id: str
    root: pathlib.Path


def _parse_dataset(value: str) -> DatasetSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected --dataset in REPO_ID=/absolute/or/relative/root format")
    repo_id, root = value.split("=", 1)
    repo_id = repo_id.strip()
    root = root.strip()
    if not repo_id:
        raise argparse.ArgumentTypeError("Dataset repo_id cannot be empty")
    if not root:
        raise argparse.ArgumentTypeError("Dataset root cannot be empty")
    return DatasetSpec(repo_id=repo_id, root=pathlib.Path(root).expanduser().resolve())


def _validate_dataset_roots(dataset_specs: Sequence[DatasetSpec]) -> None:
    if not dataset_specs:
        raise ValueError("At least one --dataset is required")
    for spec in dataset_specs:
        info_path = spec.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Local LeRobot dataset root does not contain meta/info.json: {spec.root}")


def _normalise_sampling_weights(
    dataset_specs: Sequence[DatasetSpec],
    sampling_weights: Sequence[float] | None,
    *,
    equal_dataset_sampling: bool,
) -> list[float] | None:
    """Return dataset-level sampling weights, or None for size-proportional sampling."""

    if equal_dataset_sampling and sampling_weights:
        raise ValueError("Use either --equal-dataset-sampling or --sampling-weight, not both")

    if equal_dataset_sampling:
        return [1.0 / len(dataset_specs)] * len(dataset_specs)

    if sampling_weights is None:
        return None

    if len(sampling_weights) != len(dataset_specs):
        raise ValueError(
            f"--sampling-weight count ({len(sampling_weights)}) must match --dataset count ({len(dataset_specs)})"
        )
    if any(weight <= 0 for weight in sampling_weights):
        raise ValueError("--sampling-weight values must be positive")

    total = float(sum(sampling_weights))
    return [float(weight) / total for weight in sampling_weights]


def _default_asset_output_dir(config_name: str, asset_id: str) -> pathlib.Path:
    return (pathlib.Path("assets") / config_name / asset_id).resolve()


def _load_train_script() -> types.ModuleType:
    train_path = pathlib.Path(__file__).with_name("train.py")
    spec = importlib.util.spec_from_file_location("_openpi_train_script", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training script from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config_with_mixed_asset(args: argparse.Namespace) -> _config.TrainConfig:
    config = _config.get_config(args.config_name)

    model = config.model
    if args.action_horizon is not None:
        model = dataclasses.replace(model, action_horizon=args.action_horizon)

    data_factory = config.data
    assets = dataclasses.replace(data_factory.assets, asset_id=args.asset_id)
    data_factory = dataclasses.replace(
        data_factory,
        # repo_id controls the loaded asset path for this mixed run. The real
        # source repo ids are preserved per DatasetSpec and passed to LeRobot.
        repo_id=args.asset_id,
        dataset_root=None,
        assets=assets,
    )

    policy_metadata = dict(config.policy_metadata or {})
    policy_metadata.update(
        {
            "mixed_asset_id": args.asset_id,
            "training_datasets": [{"repo_id": spec.repo_id, "root": str(spec.root)} for spec in args.dataset],
            "sampling": (
                "size_proportional"
                if args.resolved_sampling_weights is None
                else {
                    spec.repo_id: weight
                    for spec, weight in zip(args.dataset, args.resolved_sampling_weights, strict=True)
                }
            ),
            "action_horizon": model.action_horizon,
        }
    )

    replace_kwargs = {
        "data": data_factory,
        "model": model,
        "policy_metadata": policy_metadata,
    }
    for field_name in (
        "exp_name",
        "batch_size",
        "num_train_steps",
        "save_interval",
        "keep_period",
        "log_interval",
        "num_workers",
        "fsdp_devices",
        "seed",
        "checkpoint_base_dir",
        "assets_base_dir",
        "project_name",
        "overwrite",
        "resume",
        "wandb_enabled",
    ):
        if hasattr(args, field_name) and getattr(args, field_name) is not None:
            replace_kwargs[field_name] = getattr(args, field_name)

    return dataclasses.replace(config, **replace_kwargs)


def _create_single_prompted_dataset(
    data_config: _config.DataConfig,
    spec: DatasetSpec,
    *,
    action_horizon: int,
):
    metadata = _data_loader.lerobot_dataset.LeRobotDatasetMetadata(spec.repo_id, root=str(spec.root))
    dataset = _data_loader.lerobot_dataset.LeRobotDataset(
        spec.repo_id,
        root=str(spec.root),
        delta_timestamps={
            key: [t / metadata.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )
    if data_config.prompt_from_task:
        dataset = _data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(metadata.tasks)])
    return dataset


def _make_weighted_sampler(
    datasets: Sequence[object],
    dataset_weights: Sequence[float],
    *,
    seed: int,
) -> torch.utils.data.WeightedRandomSampler:
    per_index_weights = []
    for dataset, dataset_weight in zip(datasets, dataset_weights, strict=True):
        length = len(dataset)  # type: ignore[arg-type]
        if length <= 0:
            raise ValueError("Cannot sample from an empty dataset")
        per_index_weights.append(np.full(length, dataset_weight / length, dtype=np.float64))

    weights = torch.as_tensor(np.concatenate(per_index_weights), dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def _create_multi_data_loader(
    config: _config.TrainConfig,
    *,
    dataset_specs: Sequence[DatasetSpec],
    sampling_weights: Sequence[float] | None,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: str = "jax",
):
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None and not skip_norm_stats:
        raise FileNotFoundError(
            "Mixed normalization stats were not found. Run this script's compute-norm subcommand first "
            f"with --asset-id {data_config.asset_id!r}."
        )

    datasets = [
        _create_single_prompted_dataset(data_config, spec, action_horizon=config.model.action_horizon)
        for spec in dataset_specs
    ]
    raw_dataset = torch.utils.data.ConcatDataset(datasets)
    transformed_dataset = _data_loader.transform_dataset(raw_dataset, data_config, skip_norm_stats=skip_norm_stats)

    sampler = None
    if sampling_weights is not None:
        sampler = _make_weighted_sampler(datasets, sampling_weights, seed=config.seed)

    if framework == "pytorch":
        if torch.distributed.is_initialized():
            raise NotImplementedError("This helper only supports single-process PyTorch sampling")
        local_batch_size = config.batch_size
    else:
        local_batch_size = config.batch_size // jax.process_count()

    logging.info(
        "Created mixed G01 data loader: %s",
        [
            {
                "repo_id": spec.repo_id,
                "root": str(spec.root),
                "num_frames": len(dataset),  # type: ignore[arg-type]
                "sampling_weight": None if sampling_weights is None else sampling_weights[index],
            }
            for index, (spec, dataset) in enumerate(zip(dataset_specs, datasets, strict=True))
        ],
    )

    torch_loader = _data_loader.TorchDataLoader(
        transformed_dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        framework=framework,
    )
    return _data_loader.DataLoaderImpl(data_config, torch_loader)


def _compute_mixed_norm_stats(args: argparse.Namespace) -> None:
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when set")

    config = _config.get_config(args.config_name)
    action_horizon = args.action_horizon if args.action_horizon is not None else config.model.action_horizon

    stats = {"state": _normalize.RunningStats(), "actions": _normalize.RunningStats()}
    total_processed = 0

    for dataset_spec in args.dataset:
        files = _fast_stats._iter_parquet_files(dataset_spec.root)
        dataset_processed = 0
        for path in tqdm.tqdm(files, desc=f"Computing stats: {dataset_spec.repo_id}"):
            raw_state, raw_action = _fast_stats._read_episode_arrays(path)
            state = _fast_stats._select_state(raw_state)
            actions = _fast_stats._select_action(raw_action)
            action_chunks = _fast_stats._apply_joint_delta(
                _fast_stats._make_action_chunks(actions, action_horizon),
                state,
            )

            if args.max_frames is not None:
                remaining = args.max_frames - total_processed
                if remaining <= 0:
                    break
                state = state[:remaining]
                action_chunks = action_chunks[:remaining]

            stats["state"].update(state)
            stats["actions"].update(action_chunks)
            total_processed += state.shape[0]
            dataset_processed += state.shape[0]

        print(f"{dataset_spec.repo_id}: processed frames={dataset_processed}, root={dataset_spec.root}")
        if args.max_frames is not None and total_processed >= args.max_frames:
            break

    if total_processed < 2:
        raise ValueError(f"Need at least 2 frames to compute stats, got {total_processed}")

    output_dir = args.output_dir or _default_asset_output_dir(args.config_name, args.asset_id)
    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    print(f"Total processed frames: {total_processed}")
    print(f"Writing mixed stats to: {output_dir}")
    _normalize.save(output_dir, norm_stats)


def _train_mixed(args: argparse.Namespace) -> None:
    config = _config_with_mixed_asset(args)
    train_script = _load_train_script()

    original_create_data_loader = _data_loader.create_data_loader

    def create_data_loader_for_mixed_g01(
        train_config: _config.TrainConfig,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        skip_norm_stats: bool = False,
        framework: str = "jax",
    ):
        return _create_multi_data_loader(
            train_config,
            dataset_specs=args.dataset,
            sampling_weights=args.resolved_sampling_weights,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )

    _data_loader.create_data_loader = create_data_loader_for_mixed_g01
    try:
        train_script.main(config)
    finally:
        _data_loader.create_data_loader = original_create_data_loader


# ==================== AgiBot G01 multi-dataset helper END ====================


def _add_shared_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset,
        required=True,
        help="Repeat as REPO_ID=LOCAL_LEROBOT_ROOT.",
    )
    parser.add_argument("--asset-id", required=True, help="Mixed asset id used under assets/<config-name>/.")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--action-horizon", type=int, default=None, help="Defaults to the selected config value.")


def _add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sampling-weight",
        action="append",
        type=float,
        default=None,
        help="Repeat once per --dataset. Values are normalized automatically.",
    )
    parser.add_argument(
        "--equal-dataset-sampling",
        action="store_true",
        help="Sample each dataset equally instead of proportional to frame count.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    norm_parser = subparsers.add_parser(
        "compute-norm",
        help="Compute one mixed norm_stats.json over all local datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared_dataset_args(norm_parser)
    norm_parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    norm_parser.add_argument("--max-frames", type=int, default=None, help="Optional deterministic prefix subset.")

    train_parser = subparsers.add_parser(
        "train",
        help="Train the selected openpi config with a mixed LeRobot data loader.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared_dataset_args(train_parser)
    _add_sampling_args(train_parser)
    train_parser.add_argument("--exp-name", required=True)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--num-train-steps", type=int, default=None)
    train_parser.add_argument("--save-interval", type=int, default=None)
    train_parser.add_argument("--keep-period", type=int, default=None)
    train_parser.add_argument("--log-interval", type=int, default=None)
    train_parser.add_argument("--num-workers", type=int, default=None)
    train_parser.add_argument("--fsdp-devices", type=int, default=None)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--checkpoint-base-dir", default=None)
    train_parser.add_argument("--assets-base-dir", default=None)
    train_parser.add_argument("--project-name", default=None)
    train_parser.add_argument("--overwrite", action="store_true", default=None)
    train_parser.add_argument("--resume", action="store_true", default=None)
    train_parser.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_dataset_roots(args.dataset)
    args.resolved_sampling_weights = None
    if args.command == "train":
        args.resolved_sampling_weights = _normalise_sampling_weights(
            args.dataset,
            args.sampling_weight,
            equal_dataset_sampling=args.equal_dataset_sampling,
        )

    if args.command == "compute-norm":
        _compute_mixed_norm_stats(args)
    elif args.command == "train":
        _train_mixed(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
