from src.camera_utils import calculate_fov


def test_calculate_fov_preserves_legacy_ratio_behavior():
    assert calculate_fov(25, 50) == 0.5
