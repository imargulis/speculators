"""Tests for robust local directory-bundle discovery in load_raw_dataset.

Covers grouping files by (format, source), concatenating heterogeneous splits on
their common columns (the failure mode behind a DatasetGenerationError on bundles
whose splits carry different metadata keys), and the load_raw_dataset directory
entry point.
"""

import json
from pathlib import Path

import pytest
from datasets import Dataset as HFDataset

from speculators.data_generation.preprocessing import (
    SOURCE_COLUMN,
    _collect_local_dataset_groups,
    _concat_with_common_columns,
    _load_local_directory,
    load_raw_dataset,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_collect_groups_by_format_and_source(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _write_jsonl(tmp_path / "a" / "x.jsonl", [{"conversations": []}])
    _write_jsonl(tmp_path / "b" / "y.jsonl", [{"conversations": []}])

    groups = _collect_local_dataset_groups(tmp_path)

    assert sorted(g.source for g in groups) == ["a", "b"]
    assert all(g.dataset_format == "json" for g in groups)


def test_collect_groups_rejects_unsupported_single_file(tmp_path: Path):
    bad = tmp_path / "data.txt"
    bad.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported local data file extension"):
        _collect_local_dataset_groups(bad)


def test_concat_with_common_columns_drops_non_shared():
    d1 = HFDataset.from_dict(
        {"conversations": [[{"role": "user", "content": "hi"}]], "idx": [1]}
    )
    d2 = HFDataset.from_dict(
        {"conversations": [[{"role": "user", "content": "yo"}]], "secondary_id": ["s"]}
    )

    combined = _concat_with_common_columns([d1, d2])

    assert combined.column_names == ["conversations"]
    assert len(combined) == 2


def test_concat_raises_without_common_columns():
    d1 = HFDataset.from_dict({"a": [1]})
    d2 = HFDataset.from_dict({"b": [2]})
    with pytest.raises(ValueError, match="no common columns"):
        _concat_with_common_columns([d1, d2])


def test_concat_single_dataset_passthrough():
    d1 = HFDataset.from_dict({"a": [1, 2]})
    assert _concat_with_common_columns([d1]) is d1


def test_load_local_directory_heterogeneous_schema(tmp_path: Path):
    # Mirrors the real nemotron bundle: splits share 'conversations' but carry
    # different metadata keys (idx vs secondary_id). A naive single load_dataset
    # over all files raises DatasetGenerationError; grouping + common-column concat
    # loads them cleanly.
    _write_jsonl(
        tmp_path / "code.jsonl",
        [{"conversations": [{"role": "user", "content": "c"}], "idx": 1}],
    )
    _write_jsonl(
        tmp_path / "chat.jsonl",
        [{"conversations": [{"role": "user", "content": "h"}], "secondary_id": "x"}],
    )

    ds = _load_local_directory(tmp_path)

    assert ds.column_names == ["conversations"]
    assert len(ds) == 2
    assert SOURCE_COLUMN not in ds.column_names


def test_load_raw_dataset_directory(tmp_path: Path):
    _write_jsonl(
        tmp_path / "a.jsonl",
        [{"conversations": [{"role": "user", "content": "hi"}]}],
    )
    _write_jsonl(
        tmp_path / "b.jsonl",
        [{"conversations": [{"role": "user", "content": "yo"}]}],
    )

    ds, normalize_fn = load_raw_dataset(str(tmp_path))

    assert normalize_fn is None
    assert len(ds) == 2
    assert "conversations" in ds.column_names


def test_load_raw_dataset_empty_directory_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="No supported data files"):
        load_raw_dataset(str(tmp_path))


def test_load_raw_dataset_disjoint_data_columns_raises(tmp_path: Path):
    # Files share no real data column (only the internal provenance column);
    # this must fail loud rather than silently producing an empty dataset.
    _write_jsonl(
        tmp_path / "a.jsonl", [{"conversations": [{"role": "user", "content": "hi"}]}]
    )
    _write_jsonl(
        tmp_path / "b.jsonl", [{"messages": [{"role": "user", "content": "yo"}]}]
    )
    with pytest.raises(ValueError, match="no common columns"):
        load_raw_dataset(str(tmp_path))
