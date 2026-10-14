from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.metadata import write_dummy_manifest


if __name__ == "__main__":
    write_dummy_manifest(
        raw_dir=Path("data/raw"),
        output_path=Path("data/metadata/dataset_manifest_dummy.csv"),
    )
