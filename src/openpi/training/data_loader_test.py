import dataclasses
from unittest import mock

import jax
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


# ==================== AgiBot G01 π0.5 adaptation: local LeRobot root tests BEGIN ====================
def test_local_dataset_root_is_forwarded(tmp_path):
    dataset_root = tmp_path / "task_5093"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text("{}")
    model_config = pi0_config.Pi0Config(action_horizon=16)
    data_config = _config.DataConfig(
        repo_id="agibot/task_5093",
        dataset_root=str(dataset_root),
        action_sequence_keys=("action",),
    )
    metadata = mock.Mock(fps=30, tasks={0: "open the door"})

    with (
        mock.patch.object(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", return_value=metadata) as meta_cls,
        mock.patch.object(_data_loader, "EpisodeFilteredLeRobotDataset", return_value=mock.Mock()) as dataset_cls,
    ):
        _data_loader.create_torch_dataset(data_config, 16, model_config)

    meta_cls.assert_called_once_with("agibot/task_5093", root=str(dataset_root))
    assert dataset_cls.call_args.kwargs["root"] == str(dataset_root)
    assert dataset_cls.call_args.kwargs["delta_timestamps"]["action"] == [index / 30 for index in range(16)]


def test_missing_local_dataset_root_fails_before_lerobot(tmp_path):
    data_config = _config.DataConfig(repo_id="agibot/task_5093", dataset_root=str(tmp_path / "missing"))
    with (
        mock.patch.object(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata") as meta_cls,
        pytest.raises(FileNotFoundError, match="meta/info.json"),
    ):
        _data_loader.create_torch_dataset(data_config, 16, pi0_config.Pi0Config(action_horizon=16))
    meta_cls.assert_not_called()


def test_required_local_dataset_root_cannot_fall_back_to_hub():
    data_config = _config.DataConfig(repo_id="agibot/task_5093", local_dataset_only=True)
    with (
        mock.patch.object(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata") as meta_cls,
        pytest.raises(ValueError, match="explicit local dataset root"),
    ):
        _data_loader.create_torch_dataset(data_config, 16, pi0_config.Pi0Config(action_horizon=16))
    meta_cls.assert_not_called()


def test_episode_exclusion_is_forwarded_to_lerobot(tmp_path):
    dataset_root = tmp_path / "task_5867_479"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text("{}")
    (dataset_root / "meta" / "episodes.jsonl").write_text(
        '{"episode_index": 0, "length": 10}\n'
        '{"episode_index": 1, "length": 10}\n'
    )
    exclusion_file = tmp_path / "exclude_g01.txt"
    exclusion_file.write_text(
        "agibot/task_5867_479\t1\tdata/chunk-000/episode_000001.parquet\tcontrol jump\n"
    )
    data_config = _config.DataConfig(
        repo_id="agibot/task_5867_479",
        dataset_root=str(dataset_root),
        exclude_file=str(exclusion_file),
        action_sequence_keys=("action",),
    )
    metadata = mock.Mock(fps=30, tasks={0: "open the cabinet"})

    with (
        mock.patch.object(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", return_value=metadata),
        mock.patch.object(_data_loader, "EpisodeFilteredLeRobotDataset", return_value=mock.Mock()) as dataset_cls,
    ):
        _data_loader.create_torch_dataset(data_config, 16, pi0_config.Pi0Config(action_horizon=16))

    assert dataset_cls.call_args.kwargs["episodes"] == [0]


def test_filtered_lerobot_dataset_remaps_original_episode_index():
    dataset = object.__new__(_data_loader.EpisodeFilteredLeRobotDataset)
    dataset._episode_positions = {0: 0, 2: 1}
    dataset.episode_data_index = {
        "from": torch.tensor([0, 10]),
        "to": torch.tensor([10, 20]),
    }
    dataset.delta_indices = {"action": [0, 1]}

    query_indices, padding = dataset._get_query_indices(idx=10, ep_idx=2)

    assert query_indices == {"action": [10, 11]}
    assert padding["action_is_pad"].tolist() == [False, False]
# ==================== AgiBot G01 π0.5 adaptation: local LeRobot root tests END ====================
