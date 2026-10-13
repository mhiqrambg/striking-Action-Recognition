import numpy as np
from pathlib import Path
from src.data.collection import save_pose_array


def test_save_pose_array(tmp_path):
    arr = np.zeros((50, 38), dtype=np.float32)
    out = save_pose_array(arr, str(tmp_path), "normal", "video1")
    assert Path(out).exists()
    loaded = np.load(out)
    assert loaded.shape == (50, 38)
    assert Path(out).name == "normal_video1.npy"
