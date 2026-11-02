"""The job runner, the resumable log and the manifests."""

import json

import pytest

from order_book import provenance
from order_book.provenance import SIMULATOR_VERSION, source_version, write_manifest
from order_book.runner import iter_jobs, load_results


def _double(x):
    return 2 * x


def test_jobs_come_back_in_order_serial_and_parallel():
    jobs = list(range(20))
    assert list(iter_jobs(_double, jobs)) == [2 * j for j in jobs]
    assert list(iter_jobs(_double, jobs, workers=2, chunksize=3)) == [2 * j for j in jobs]


def test_load_results_survives_a_job_killed_mid_write(tmp_path):
    path = tmp_path / "log.jsonl"
    assert load_results(path) == []
    path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3')           # killed half way through the third record
    assert load_results(path) == [{"a": 1}, {"a": 2}]
    assert path.read_text() == '{"a": 1}\n{"a": 2}\n'         # safe to append to again
    assert load_results(path, lambda r: r["a"]) == [1, 2]


def test_manifest_records_the_run_and_a_fresh_run_replaces_it(tmp_path):
    write_manifest(tmp_path, {"seeds": [1, 2]})
    entries = json.loads((tmp_path / "manifest.json").read_text())
    assert len(entries) == 1 and entries[0]["simulator_version"] == SIMULATOR_VERSION
    assert entries[0]["config"] == {"seeds": [1, 2]} and entries[0]["numpy"] and "argv" in entries[0]
    write_manifest(tmp_path, {"seeds": [3]}, resume=True)
    entries = json.loads((tmp_path / "manifest.json").read_text())
    assert len(entries) == 2 and entries[1]["resumed"] is True
    write_manifest(tmp_path, {"seeds": [4]})
    assert len(json.loads((tmp_path / "manifest.json").read_text())) == 1


def test_resuming_across_simulator_versions_is_refused(tmp_path, monkeypatch):
    write_manifest(tmp_path, {})
    monkeypatch.setattr(provenance, "SIMULATOR_VERSION", "not-this-one")
    with pytest.raises(RuntimeError, match="mix two markets"):
        write_manifest(tmp_path, {}, resume=True)
    write_manifest(tmp_path, {})                              # starting over is fine


def test_source_version_falls_back_to_what_the_sync_recorded(tmp_path, monkeypatch):
    """The cluster has no .git, so the commit comes from the file hpc/push.sh writes."""
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    monkeypatch.setattr(provenance, "SOURCE_FILE", tmp_path / ".source_version")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert source_version()["commit"] is None
    (tmp_path / ".source_version").write_text(json.dumps({"commit": "abc123", "dirty": True}))
    version = source_version()
    assert version["commit"] == "abc123" and version["dirty"] is True and version["origin"] == ".source_version"


def test_a_shard_is_one_seed_and_out_of_range_is_refused():
    from order_book.runner import seed_of_shard

    assert seed_of_shard((1000, 1001, 1002), 1) == 1000 and seed_of_shard((1000, 1001, 1002), 3) == 1002
    with pytest.raises(ValueError):
        seed_of_shard((1000, 1001), 3)


def test_missing_shards_names_only_those_with_work_left():
    from order_book.runner import missing_shards

    expected = {("a", 1): 1, ("b", 1): 1, ("a", 2): 2, ("a", 3): 3}
    assert missing_shards(expected, {("a", 1), ("a", 2)}, (1, 2, 3)) == [1, 3]
    assert missing_shards(expected, set(expected), (1, 2, 3)) == []


def test_shards_from_different_simulator_versions_cannot_be_merged(tmp_path, monkeypatch):
    from order_book.runner import load_shards, shard_dir

    write_manifest(shard_dir(tmp_path, 1), {})
    monkeypatch.setattr(provenance, "SIMULATOR_VERSION", "other")
    write_manifest(shard_dir(tmp_path, 2), {})
    with pytest.raises(RuntimeError, match="cannot be merged"):
        load_shards(tmp_path)
