import os
import subprocess
import argparse
import cv2
import shutil
from pathlib import Path

def extract_frames(video_path, images_dir, extract_fps=5):
    """
    동영상 파일에서 지정된 프레임 레이트(fps)로 이미지를 추출합니다.
    """
    print(f"[LOG] 영상에서 프레임 추출을 시작합니다: {video_path}")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(original_fps / extract_fps))

    count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_name = os.path.join(images_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"[LOG] 총 {saved_count}장의 이미지가 추출되었습니다: {images_dir}")

def run_colmap(workspace_dir):
    """
    Colmap 파이프라인(Feature Extraction -> Matching -> Mapping)을 자동 실행합니다.
    """
    print("\n[LOG] Colmap 파이프라인을 시작합니다...")
    
    colmap_exe = "colmap" # 환경변수에 등록된 colmap 명령어 (Windows의 경우 colmap.bat 일 수 있음)
    db_path = os.path.join(workspace_dir, "database.db")
    images_dir = os.path.join(workspace_dir, "images")
    sparse_dir = os.path.join(workspace_dir, "sparse")

    # 기존 DB가 있다면 삭제 (충돌 방지)
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(sparse_dir):
        shutil.rmtree(sparse_dir)
    os.makedirs(sparse_dir)

    # 1. Feature Extraction (특징점 추출)
    # 2번 이슈의 목표인 왜곡 보정을 위해 카메라 모델을 OPENCV로 설정합니다.
    print("\n--- 1. Feature Extraction ---")
    subprocess.run([
        colmap_exe, "feature_extractor",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", "1" # 모든 이미지가 같은 카메라로 촬영되었다고 가정
    ], check=True)

    # 2. Exhaustive Matching (특징점 매칭)
    # 소형 제품 스캔은 이미지가 서로 오버랩되는 경우가 많으므로 Exhaustive 매칭이 유리합니다.
    print("\n--- 2. Feature Matching ---")
    subprocess.run([
        colmap_exe, "exhaustive_matcher",
        "--database_path", db_path
    ], check=True)

    # 3. Mapper (3D 맵핑 및 카메라 포즈 계산)
    print("\n--- 3. Mapping (SfM) ---")
    subprocess.run([
        colmap_exe, "mapper",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--output_path", sparse_dir
    ], check=True)

    print(f"\n[LOG] Colmap 파이프라인 완료! 결과물이 다음 경로에 저장되었습니다: {sparse_dir}/0/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="소형 제품 3D 스캔용 Colmap 자동화 파이프라인")
    parser.add_argument("--video", type=str, required=True, help="입력 동영상 파일 경로 (.mp4 등)")
    parser.add_argument("--workspace", type=str, default="./workspace", help="Colmap 작업 및 결과물이 저장될 작업 폴더")
    parser.add_argument("--fps", type=int, default=5, help="1초당 추출할 프레임 수 (기본값: 5)")
    
    args = parser.parse_args()

    # 작업 공간 세팅
    workspace = Path(args.workspace)
    images_folder = workspace / "images"
    workspace.mkdir(parents=True, exist_ok=True)

    # 1. 영상에서 이미지 추출
    extract_frames(args.video, str(images_folder), extract_fps=args.fps)

    # 2. Colmap 실행
    try:
        run_colmap(str(workspace))
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Colmap 실행 중 오류가 발생했습니다. 환경변수 설정이나 이미지 상태를 확인해주세요. (Error: {e})")
    except FileNotFoundError:
        print("\n[ERROR] 'colmap' 명령어를 찾을 수 없습니다. Colmap이 설치되어 있고 환경 변수 PATH에 등록되어 있는지 확인하세요.")