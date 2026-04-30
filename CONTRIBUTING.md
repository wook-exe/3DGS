# 🤝 기여 가이드 (Contributing to 3DGS-Viewer)

3DGS-Viewer 프로젝트에 관심을 가져주셔서 감사합니다! 이 문서는 프로젝트에 기여하기 위한 가이드라인입니다.

## 1. 이슈(Issue) 등록
버그를 발견하셨거나 새로운 기능을 제안하고 싶으시다면, 먼저 GitHub Issues 탭에서 이슈를 생성해 주세요.

## 2. 브랜치 전략 (Branching)
작업을 시작하기 전, 반드시 `main` 브랜치에서 새로운 브랜치를 생성해 주세요.
- **기능 추가:** `feature/이슈번호-작업내용` (예: `feature/10-add-threejs-viewer`)
- **버그 수정:** `bugfix/이슈번호-버그내용` (예: `bugfix/12-fix-memory-leak`)

## 3. 커밋 컨벤션 (Conventional Commits)
명확한 히스토리 관리를 위해 아래의 커밋 메시지 규칙을 지켜주세요.
- `feat:` 새로운 기능 추가
- `fix:` 버그 수정
- `docs:` 문서 수정
- `refactor:` 코드 리팩토링 (기능 변화 없음)

## 4. Pull Request (PR) 가이드
- 작업이 완료되면 `main` 브랜치로 PR을 생성해 주세요.
- PR 제목은 커밋 메시지 규칙과 동일하게 작성합니다.
- 리뷰어(Reviewer)를 지정하고, 최소 1명 이상의 Approve를 받아야 Merge가 가능합니다.
