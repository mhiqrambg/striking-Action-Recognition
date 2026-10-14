from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


LABEL_TO_DUMMY_KSS = {"normal": 3, "light": 5, "heavy": 8}


def build_dummy_manifest(raw_dir: Path) -> List[Dict[str, str]]:
    """Index source videos with development-only KSS placeholder labels."""
    records: List[Dict[str, str]] = []
    for label, kss_score in LABEL_TO_DUMMY_KSS.items():
        for index, video_path in enumerate(sorted((raw_dir / label).glob("*.mp4")), start=1):
            if "overlay" in video_path.stem.lower():
                continue
            records.append(
                {
                    "video_id": f"{label}_{index:03d}",
                    "video_path": video_path.as_posix(),
                    "label": label,
                    "class_id": str(list(LABEL_TO_DUMMY_KSS).index(label)),
                    "subject_id": "PSEUDO_GROUP_NOT_A_PARTICIPANT",
                    "session_id": "DUMMY_SESSION",
                    "kss_score": str(kss_score),
                    "label_source": "DUMMY_FOR_DEVELOPMENT",
                }
            )
    return records


def write_dummy_manifest(raw_dir: Path, output_path: Path) -> None:
    records = build_dummy_manifest(raw_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
