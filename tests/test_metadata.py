from pathlib import Path
import subprocess
import sys

from src.data.metadata import build_dummy_manifest


def test_build_dummy_manifest_indexes_each_source_video_once():
    records = build_dummy_manifest(Path("data/raw"))

    assert len(records) == 45
    assert {record["label"] for record in records} == {"normal", "light", "heavy"}
    assert {record["label_source"] for record in records} == {"DUMMY_FOR_DEVELOPMENT"}
    assert all(record["video_path"].endswith(".mp4") for record in records)
    assert all("overlay" not in record["video_path"] for record in records)


def test_build_dummy_manifest_uses_documented_kss_mapping():
    records = build_dummy_manifest(Path("data/raw"))
    kss_by_label = {record["label"]: record["kss_score"] for record in records}

    assert kss_by_label == {"normal": "3", "light": "5", "heavy": "8"}


def test_create_dummy_manifest_script_runs_from_repository_root():
    result = subprocess.run(
        [sys.executable, "scripts/create_dummy_manifest.py"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
