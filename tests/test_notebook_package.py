import json
from pathlib import Path


EXPECTED_NOTEBOOKS = [
    "00_index.ipynb",
    "01_environment_and_reproducibility.ipynb",
    "02_data_inventory_and_labels.ipynb",
    "03_pose_and_feature_extraction.ipynb",
    "04_sequence_dataset_creation.ipynb",
    "05_model_architecture.ipynb",
    "06_train_A_baseline.ipynb",
    "07_train_B_proposed.ipynb",
    "08_train_C_D_ablation.ipynb",
    "09_cross_validation_and_evaluation.ipynb",
    "10_results_and_hypothesis_testing.ipynb",
    "11_realtime_benchmark.ipynb",
    "12_realtime_demo.ipynb",
]


def test_thesis_notebook_package_is_complete_and_identifies_dummy_labels():
    notebook_dir = Path("notebooks")
    assert [path.name for path in sorted(notebook_dir.glob("*.ipynb"))] == EXPECTED_NOTEBOOKS

    for filename in EXPECTED_NOTEBOOKS:
        notebook = json.loads((notebook_dir / filename).read_text())
        assert notebook["nbformat"] == 4
        assert notebook["cells"]

    data_notebook = (notebook_dir / "02_data_inventory_and_labels.ipynb").read_text()
    assert "DUMMY_FOR_DEVELOPMENT" in data_notebook
    assert "bukan ground truth tesis final" in data_notebook


def test_notebook_readme_documents_execution_order():
    readme = Path("notebooks/README.md").read_text()
    assert "00_index.ipynb" in readme
    assert "12_realtime_demo.ipynb" in readme
    assert "dataset_manifest_dummy.csv" in readme
