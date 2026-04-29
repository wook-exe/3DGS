***

# Hi there! 👋

인공지능을 전공하고 있는 4학년 개발자입니다. AI 기술과 실제 사용자 경험을 연결하는 직관적인 소프트웨어를 만드는 데 큰 흥미를 가지고 있습니다. 최근에는 복잡한 3D 객체를 실시간으로 렌더링하고 상호작용할 수 있는 3D 컴퓨터 비전 기술에 집중하고 있습니다.

### 🔭 Areas of Interest (관심 분야)
*   **3D Computer Vision:** NeRF, 3D Gaussian Splatting(3DGS) 등 최신 기술을 활용한 고품질 3D 장면 생성 및 사실적인 시점 합성
*   **AI Content Generation:** 인공지능 기반의 비디오 및 인터랙티브 콘텐츠 생성 파이프라인
*   **Cross-Platform Application:** AI 모델의 결과를 사용자가 쉽게 경험할 수 있도록 돕는 빠르고 안정적인 앱/웹 서비스 구축

### 🌱 Learning Goals (학습 목표)
*   **3D 렌더링 파이프라인 최적화:** Colmap(SfM)을 이용한 정확한 카메라 포즈 추정부터 3DGS 파라미터 최적화까지, 전체 프로세스를 깊이 있게 이해하고 성능 끌어올리기
*   **컴퓨터 비전 알고리즘 고도화:** Python과 C/C++을 적재적소에 활용하여 핵심 렌더링 알고리즘의 처리 속도와 자원 효율성 향상
*   **사용자 중심의 인터랙션 구현:** WebGL, Three.js 등을 활용해 웹 브라우저 및 HMD 환경에서 지연 없는 부드러운 자유 시점 조작(마우스, Head Tracking) 인터페이스 마스터하기

### 🚀 Current Project
**소형 제품 3D 스캔 및 360° 가상 시점 뷰어 개발**
*   고가의 스캐너 없이 스마트폰 영상만으로 제품의 형태와 색상을 완벽하게 반영하는 3DGS 모델 생성 
*   웹 및 HMD 환경에서 사용자가 자유롭게 시점을 이동하며 관찰할 수 있는 실시간 뷰어 엔진 개발

### 💻 Tech Stack
*   **Languages:** Python, C/C++
*   **Frameworks & Libraries:** Flutter, Firebase, OpenCV
*   **3D Vision:** Colmap, NeRF, 3DGS

<!-- 기존 내용 생략 -->

### 🚀 Current Project
**소형 제품 3D 스캔 및 360° 가상 시점 뷰어 개발**
*   고가의 스캐너 없이 스마트폰 영상만으로 제품의 형태와 색상을 완벽하게 반영하는 3DGS 모델 생성 
*   웹 및 HMD 환경에서 사용자가 자유롭게 시점을 이동하며 관찰할 수 있는 실시간 뷰어 엔진 개발

### 💻 Tech Stack
*   **Languages:** Python, C/C++
*   **Frameworks & Libraries:** Flutter, Firebase, OpenCV
*   **3D Vision:** Colmap, NeRF, 3DGS
*   **DevOps & Monitoring:** GitHub Actions, Chart.js

---

### 📊 Project Performance Metrics (DORA)
성공적인 프로젝트 운영과 지속적인 통합/배포(CI/CD) 품질을 측정하기 위해 **DORA 4대 지표**를 GitHub Actions로 자동 수집하여 모니터링하고 있습니다.

![DORA Metrics Dashboard](./assets/dora-dashboard.png)

*   **Deployment Frequency:** 주간 배포 횟수 모니터링
*   **Lead Time for Changes:** 커밋부터 프로덕션 배포까지의 소요 시간
*   **MTTR:** 장애 발생(Incident) 시 평균 복구 시간
*   **Change Failure Rate:** 전체 배포 중 결함이 발생한 비율


***

# 📸 3DGS-Viewer (소형 제품 3D 스캔 및 360° 가상 시점 뷰어)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![C++](https://img.shields.io/badge/C%2B%2B-14%2B-orange)

## 📖 개요 (Overview)
**3DGS-Viewer**는 고가의 3D 스캐너 없이, 스마트폰이나 일반 카메라로 촬영한 소형 제품의 영상을 기반으로 부드럽고 사실적인 3D 가상 시점을 제공하는 프로젝트입니다. 
최신 **3D Gaussian Splatting (3DGS)** 기술을 적용하여, 복잡한 메쉬(Mesh) 생성 없이 실시간 렌더링과 자유로운 시점 조작(웹/HMD)을 지원합니다.

## ✨ 주요 기능 (Features)
- 📷 **카메라 포즈 추정:** Colmap(SfM)을 활용한 입력 영상의 정밀한 카메라 위치 및 방향 계산
- 🌌 **3D Gaussian 최적화:** 제품의 형태와 색상을 완벽하게 반영하는 3DGS 모델 생성 (CUDA 가속)
- 🖥️ **가상 시점 렌더링:** WebGL 기반으로 사용자가 웹에서 자유롭게 회전/확대/축소하며 관찰 가능

## 🚀 시작하기 (Getting Started)
프로젝트 로컬 환경 설정 및 실행 방법은 [Wiki: Getting Started](https://github.com/본인아이디/레포지토리이름/wiki/Getting-Started) 문서를 참고해 주세요.

## 🤝 기여하기 (Contributing)
오픈소스 기여에 관심이 있으시다면 [CONTRIBUTING.md](./CONTRIBUTING.md)를 먼저 읽어주세요. 모든 버그 리포트, 기능 제안, PR을 환영합니다!

## 📜 라이선스 (License)
이 프로젝트는 [MIT License](./LICENSE)를 따릅니다.
