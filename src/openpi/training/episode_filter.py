"""Episode exclusion helpers shared by G01 preflight, stats, and training."""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import json
import pathlib
import re

_EPISODE_PATTERN = re.compile(r"episode_(\d+)\.parquet$")


@dataclasses.dataclass(frozen=True)
class EpisodeExclusion:
    """One episode excluded by a data-quality report or a manual decision."""

    dataset_id: str | None
    episode_index: int
    relative_path: str
    reason: str


def episode_index_from_path(path: str | pathlib.Path) -> int:
    """Extract the integer episode id from a LeRobot parquet path."""

    match = _EPISODE_PATTERN.search(pathlib.PurePosixPath(str(path)).name)
    if match is None:
        raise ValueError(f"Could not parse an episode index from {path!s}")
    return int(match.group(1))


def load_exclusions(path: str | pathlib.Path | None) -> list[EpisodeExclusion]:
    """Load the tab-separated exclusion format emitted by the G01 preflight.

    The preferred format is::

        dataset_id<TAB>episode_index<TAB>relative_parquet_path<TAB>reason

    For compatibility with early manual lists, both a relative parquet path on
    its own and ``dataset_id<TAB>relative_path<TAB>reason`` are accepted.
    """

    if path is None:
        return []

    exclusion_path = pathlib.Path(path).expanduser().resolve()
    if not exclusion_path.is_file():
        raise FileNotFoundError(f"Episode exclusion file does not exist: {exclusion_path}")

    exclusions: list[EpisodeExclusion] = []
    for line_number, raw_line in enumerate(exclusion_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = [part.strip() for part in raw_line.split("\t")]
        try:
            if len(parts) == 1:
                relative_path = parts[0]
                exclusions.append(
                    EpisodeExclusion(
                        dataset_id=None,
                        episode_index=episode_index_from_path(relative_path),
                        relative_path=relative_path,
                        reason="manual exclusion",
                    )
                )
            elif len(parts) >= 4 and parts[1].isdigit():
                exclusions.append(
                    EpisodeExclusion(
                        dataset_id=parts[0] or None,
                        episode_index=int(parts[1]),
                        relative_path=parts[2],
                        reason="\t".join(parts[3:]) or "manual exclusion",
                    )
                )
            elif len(parts) >= 2:
                relative_path = parts[1]
                exclusions.append(
                    EpisodeExclusion(
                        dataset_id=parts[0] or None,
                        episode_index=episode_index_from_path(relative_path),
                        relative_path=relative_path,
                        reason="\t".join(parts[2:]) or "manual exclusion",
                    )
                )
            else:
                raise ValueError("unsupported exclusion row")
        except ValueError as error:
            raise ValueError(f"Invalid exclusion row {line_number} in {exclusion_path}: {raw_line!r}") from error

    return exclusions


def _dataset_aliases(repo_id: str, dataset_root: str | pathlib.Path) -> set[str]:
    root = pathlib.Path(dataset_root).expanduser().resolve()
    return {
        repo_id,
        repo_id.rsplit("/", 1)[-1],
        root.name,
        str(root),
        root.as_posix(),
    }


def matching_exclusions(
    exclusions: Iterable[EpisodeExclusion],
    *,
    repo_id: str,
    dataset_root: str | pathlib.Path,
) -> dict[int, EpisodeExclusion]:
    """Return exclusions applying to one local dataset, keyed by episode id."""

    aliases = _dataset_aliases(repo_id, dataset_root)
    matches: dict[int, EpisodeExclusion] = {}
    for exclusion in exclusions:
        if (
            exclusion.dataset_id is not None
            and exclusion.dataset_id not in aliases
            and exclusion.dataset_id.rsplit("/", 1)[-1] not in aliases
        ):
            continue
        matches[exclusion.episode_index] = exclusion
    return matches


def read_episode_indices(dataset_root: str | pathlib.Path) -> list[int]:
    """Read and validate available episode ids from ``meta/episodes.jsonl``."""

    root = pathlib.Path(dataset_root).expanduser().resolve()
    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Local LeRobot dataset is missing {episodes_path}")

    episode_indices: list[int] = []
    for line_number, line in enumerate(episodes_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            episode_indices.append(int(json.loads(line)["episode_index"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid episode metadata at {episodes_path}:{line_number}") from error

    if not episode_indices:
        raise ValueError(f"No episodes are declared in {episodes_path}")
    if len(episode_indices) != len(set(episode_indices)):
        raise ValueError(f"Duplicate episode ids are declared in {episodes_path}")
    return sorted(episode_indices)


def included_episode_indices(
    dataset_root: str | pathlib.Path,
    *,
    repo_id: str,
    exclusion_file: str | pathlib.Path | None,
) -> tuple[list[int], dict[int, EpisodeExclusion]]:
    """Resolve the episodes retained for one dataset and the applied exclusions."""

    all_indices = read_episode_indices(dataset_root)
    matched = matching_exclusions(
        load_exclusions(exclusion_file),
        repo_id=repo_id,
        dataset_root=dataset_root,
    )
    unknown = sorted(set(matched).difference(all_indices))
    if unknown:
        raise ValueError(f"Exclusion file references episode ids not present in {dataset_root}: {unknown}")
    included = [episode_index for episode_index in all_indices if episode_index not in matched]
    if not included:
        raise ValueError(f"Episode exclusion removed every episode from {dataset_root}")
    return included, matched


def filter_parquet_files(
    files: Iterable[pathlib.Path],
    *,
    repo_id: str,
    dataset_root: str | pathlib.Path,
    exclusion_file: str | pathlib.Path | None,
) -> tuple[list[pathlib.Path], dict[int, EpisodeExclusion]]:
    """Filter episode parquet files with the same rules used by training."""

    included, matched = included_episode_indices(
        dataset_root,
        repo_id=repo_id,
        exclusion_file=exclusion_file,
    )
    included_set = set(included)
    filtered = [path for path in files if episode_index_from_path(path) in included_set]
    if not filtered:
        raise ValueError(f"Episode exclusion left no parquet files under {dataset_root}")
    return filtered, matched
