import numpy as np
from src.data.video_pose import normalize_and_featurize, FEATURE_DIM, NUM_KEYPOINTS


def test_feature_dim():
    assert FEATURE_DIM == 38
    assert NUM_KEYPOINTS == 11


def test_normalize_and_featurize_shape():
    k = np.zeros((17, 3), dtype=np.float32)
    k[:, 2] = 1.0
    k[0] = [0.5, 0.3, 1.0]    # nose
    k[5] = [0.4, 0.6, 1.0]    # left shoulder
    k[6] = [0.6, 0.6, 1.0]    # right shoulder
    out = normalize_and_featurize(k)
    assert out.shape == (FEATURE_DIM,)


def test_origin_is_shoulder_midpoint():
    k = np.zeros((17, 3), dtype=np.float32)
    k[:, 2] = 1.0
    k[5] = [0.4, 0.6, 1.0]
    k[6] = [0.6, 0.6, 1.0]
    k[0] = [0.5, 0.2, 1.0]    # nose tepat di atas midpoint bahu
    out = normalize_and_featurize(k)
    # x nose setelah normalisasi = 0.5 - 0.5 = 0
    assert abs(out[0] - 0.0) < 1e-5
