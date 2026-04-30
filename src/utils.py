def calculate_frame_count(duration_seconds, fps=30):
    """
    동영상의 길이와 초당 프레임(fps)을 받아 총 프레임 수를 계산합니다.
    (3DGS용 프레임 추출 전처리 단계에서 사용)
    """
    if duration_seconds < 0 or fps <= 0:
        raise ValueError("시간과 프레임은 0보다 커야 합니다.")
    return int(duration_seconds * fps)