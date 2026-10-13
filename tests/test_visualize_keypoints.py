import numpy as np

from src.data.visualize_keypoints import (
    KEYPOINT_NAMES,
    SKELETON,
    FEATURE_NAMES,
    draw_upper_body,
    draw_feature_panel,
)


def _fake_kpts():
    xy = np.random.rand(17, 2).astype(np.float32)
    conf = np.full(17, 0.9, dtype=np.float32)
    return xy, conf


def test_eleven_keypoints_defined():
    assert set(KEYPOINT_NAMES.keys()) == set(range(11))
    assert KEYPOINT_NAMES[0] == "nose"
    assert KEYPOINT_NAMES[5] == "left_shoulder"
    assert KEYPOINT_NAMES[6] == "right_shoulder"


def test_skeleton_only_upper_body():
    for a, b in SKELETON:
        assert 0 <= a <= 10 and 0 <= b <= 10


def test_five_features_defined():
    assert len(FEATURE_NAMES) == 5


def test_draw_upper_body_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    xy, conf = _fake_kpts()
    out = draw_upper_body(frame, xy, conf)
    assert out.shape == (480, 640, 3)
    assert out.dtype == np.uint8
    assert out.any()  # sesuatu tergambar


def test_draw_feature_panel_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    feat = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    out = draw_feature_panel(frame, feat)
    assert out.shape == (480, 640, 3)
    assert out.any()
