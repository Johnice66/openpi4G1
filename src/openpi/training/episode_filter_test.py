import pathlib

import pytest

import openpi.training.episode_filter as episode_filter


def _write_episodes(root: pathlib.Path, episode_indices: list[int]) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "episodes.jsonl").write_text(
        "".join(f'{{"episode_index": {episode_index}, "length": 10}}\n' for episode_index in episode_indices)
    )


def test_preflight_exclusion_format_filters_matching_dataset(tmp_path):
    root = tmp_path / "task_5867_479"
    _write_episodes(root, [0, 1, 355])
    exclusion_file = tmp_path / "exclude_g01.txt"
    exclusion_file.write_text(
        "# dataset_id<TAB>episode_index<TAB>relative_path<TAB>reason\n"
        "agibot/task_5867_479\t355\tdata/chunk-000/episode_000355.parquet\tcontrol jump\n"
        "agibot/another_task\t1\tdata/chunk-000/episode_000001.parquet\tunrelated\n"
    )

    included, exclusions = episode_filter.included_episode_indices(
        root,
        repo_id="agibot/task_5867_479",
        exclusion_file=exclusion_file,
    )

    assert included == [0, 1]
    assert list(exclusions) == [355]
    assert exclusions[355].reason == "control jump"


def test_dataset_basename_matches_when_asset_id_is_custom(tmp_path):
    root = tmp_path / "task_5867_479"
    _write_episodes(root, [0, 355])
    exclusion_file = tmp_path / "exclude_g01.txt"
    exclusion_file.write_text("agibot/task_5867_479\t355\tdata/chunk-000/episode_000355.parquet\tcontrol jump\n")

    included, _ = episode_filter.included_episode_indices(
        root,
        repo_id="agibot/custom_norm_asset",
        exclusion_file=exclusion_file,
    )

    assert included == [0]


def test_legacy_relative_path_exclusion_is_supported(tmp_path):
    root = tmp_path / "task_5867"
    _write_episodes(root, [0, 1])
    exclusion_file = tmp_path / "exclude_g01.txt"
    exclusion_file.write_text("data/chunk-000/episode_000001.parquet\n")

    included, _ = episode_filter.included_episode_indices(
        root,
        repo_id="agibot/task_5867",
        exclusion_file=exclusion_file,
    )

    assert included == [0]


def test_unknown_excluded_episode_is_rejected(tmp_path):
    root = tmp_path / "task_5867"
    _write_episodes(root, [0])
    exclusion_file = tmp_path / "exclude_g01.txt"
    exclusion_file.write_text("data/chunk-000/episode_000999.parquet\n")

    with pytest.raises(ValueError, match="not present"):
        episode_filter.included_episode_indices(
            root,
            repo_id="agibot/task_5867",
            exclusion_file=exclusion_file,
        )
