# src/camera_utils.py

def calculate_fov(focal_length, sensor_size):
    """카메라의 초점 거리와 센서 크기를 기반으로 시야각(FOV)을 계산합니다."""
    return focal_length / sensor_size