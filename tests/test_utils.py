import pytest
from src.utils import calculate_frame_count

def test_calculate_frame_count():
    # 정상 케이스 테스트: 10초짜리 30fps 영상은 300프레임이어야 함
    assert calculate_frame_count(10, 30) == 300
    
    # 60fps 테스트
    assert calculate_frame_count(5, 60) == 300

def test_calculate_frame_count_invalid_input():
    # 예외 처리(에러)가 잘 발생하는지 테스트
    with pytest.raises(ValueError):
        calculate_frame_count(-5, 30)