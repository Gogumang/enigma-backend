# Enigma

로맨스 스캠 예방을 위한 백엔드 API 서버

## 주요 기능

### 딥페이크 검사기
- **이미지 탐지**: EfficientViT + Sightengine + CLIP 앙상블
- **비디오 탐지**: Cross-EfficientViT + Sightengine 앙상블
- **AI 생성 이미지 탐지**: DALL-E, Midjourney, Stable Diffusion 등 감지
- 의심 영역 좌표 및 히트맵 제공

### 대화 분석
- AI 기반 로맨스 스캠 패턴 분석
- 스크린샷 분석 지원
- **Qdrant 벡터 검색** 기반 의미적 유사도 매칭

### 스캐머 네트워크 분석
- **Neo4j 그래프 DB** 기반 스캐머 관계 추적
- 같은 계좌/전화번호/프로필 사진 사용 스캐머 연결
- 스캠 조직 네트워크 시각화
- 연관 스캐머 자동 탐지

### 면역 훈련
- 로맨스 스캠 대응 시뮬레이션 훈련
- 5턴 대화 기반 훈련 시스템

### 프로필 검색
- 얼굴 인식 기반 스캐머 DB 검색 (DeepFace)
- 역이미지 검색 링크 제공

### URL 검사
- 피싱 URL 탐지

## 기술 스택

- **Framework**: FastAPI
- **Architecture**: DDD (Domain-Driven Design)
- **AI Services**: OpenAI GPT-4o, Sightengine, CLIP
- **Deepfake Detection**: EfficientViT, Cross-EfficientViT
- **Face Recognition**: DeepFace (VGG-Face)
- **Vector DB**: Qdrant (대화 패턴 의미 검색)
- **Graph DB**: Neo4j (스캐머 네트워크 분석)

## 탐지 모델

### 이미지 딥페이크 탐지 앙상블
| 모델 | 역할 |
|------|------|
| EfficientViT | 딥페이크 탐지 + 히트맵 |
| Sightengine (deepfake) | 얼굴 조작 탐지 |
| Sightengine (genai) | AI 생성 이미지 탐지 |
| CLIP (ViT-B/32) | AI 생성 이미지 탐지 |

### 비디오 딥페이크 탐지 앙상블
| 모델 | 역할 |
|------|------|
| Cross-EfficientViT | 비디오 딥페이크 (DFDC AUC 0.951) |
| Sightengine | 비디오 딥페이크 탐지 |

## 설치

### 1. 의존성 설치

```bash
# uv 사용 (권장)
uv sync
uv pip install git+https://github.com/openai/CLIP.git

# 또는 pip 사용
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install git+https://github.com/openai/CLIP.git
```

### 2. 환경 변수 설정

`.env` 파일을 **프로젝트 루트**에 생성:

```bash
# .env
PORT=4000
CORS_ORIGIN=*

# Sightengine (딥페이크 탐지) - 필수
SIGHTENGINE_API_USER=your_api_user
SIGHTENGINE_API_SECRET=your_api_secret

# OpenAI
OPENAI_API_KEY=your_openai_key

# Qdrant (대화 패턴 벡터 검색)
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# Neo4j (스캐머 네트워크 분석)
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# SerpApi (역이미지 검색)
SERPAPI_KEY=your_serpapi_key
```

### 3. 모델 가중치 다운로드 (선택)

```bash
# CLIP 모델 (자동 다운로드, SSL 에러 시 수동)
mkdir -p ~/.cache/clip
curl -L -k -o ~/.cache/clip/ViT-B-32.pt \
  "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"

# Cross-EfficientViT 모델 (Google Drive에서 다운로드)
# https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1
mkdir -p src/infrastructure/ai/deepfake_explainer/models
mv ~/Downloads/cross_efficient_vit.pth src/infrastructure/ai/deepfake_explainer/models/
```

## 실행 방법

### 방법 1: 가상환경 활성화 후 실행 (권장)

```bash
# 프로젝트 루트에서 실행
source .venv/bin/activate
python -m src.main
```

### 방법 2: uvicorn 직접 실행

```bash
source .venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 4000
```

### 방법 3: 환경변수 직접 지정

```bash
source .venv/bin/activate
PORT=4000 \
SIGHTENGINE_API_USER=xxx \
SIGHTENGINE_API_SECRET=xxx \
uvicorn src.main:app --host 0.0.0.0 --port 4000
```

> **주의**: `uv run python -m src.main`은 `.env` 파일을 제대로 로드하지 못할 수 있습니다.
> 반드시 `source .venv/bin/activate` 후 실행하세요.

## 접속 확인

서버 실행 후:
- API 서버: http://localhost:4000
- API 문서 (Swagger): http://localhost:4000/api/docs
- 헬스체크: http://localhost:4000/api/health

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/deepfake/analyze/image` | 이미지 딥페이크 분석 |
| POST | `/api/deepfake/analyze/video` | 영상 딥페이크 분석 |
| POST | `/api/deepfake/analyze/url` | URL 이미지 딥페이크 분석 |
| POST | `/api/chat/analyze` | 대화 분석 |
| POST | `/api/chat/analyze-screenshot` | 스크린샷 분석 |
| POST | `/api/training/start` | 면역 훈련 시작 |
| POST | `/api/training/message` | 면역 훈련 메시지 |
| POST | `/api/network/report` | 스캐머 신고 |
| GET | `/api/network/analyze/{id}` | 스캐머 네트워크 분석 |
| POST | `/api/profile/search` | 프로필 검색 |
| POST | `/api/url/check` | URL 검사 |

## 사용 예시

### 이미지 분석

```bash
curl -X POST "http://localhost:4000/api/deepfake/analyze/image" \
  -F "file=@image.jpg"
```

### 비디오 분석

```bash
curl -X POST "http://localhost:4000/api/deepfake/analyze/video" \
  -F "file=@video.mp4"
```

## Docker 배포

```bash
# 빌드 및 실행
docker-compose -f docker-compose.prod.yml up -d --build

# 로그 확인
docker-compose -f docker-compose.prod.yml logs -f
```

## 문제 해결

### "시뮬레이션 결과입니다" 메시지가 나올 때

Sightengine API 키가 로드되지 않았습니다.

**원인:**
- `.env` 파일이 없는 디렉토리에서 서버를 시작함
- 서버 시작 후 `.env` 파일을 수정함 (설정이 캐시되어 반영 안됨)
- 환경변수가 제대로 설정되지 않음

**해결 방법:**

```bash
# 1. 기존 서버 모두 종료
pkill -9 -f "uvicorn|python.*src.main"

# 또는 포트로 직접 종료
lsof -ti :4000 | xargs kill -9 2>/dev/null

# 2. 프로젝트 루트로 이동 (.env 파일이 있는 곳)
cd /path/to/deepfake-detector-py

# 3. .env 파일 확인
cat .env | grep SIGHTENGINE

# 4. 서버 재시작
uv run python -m src.main
```

**주의사항:**
- pydantic-settings는 `@lru_cache`로 설정을 캐시함
- `.env` 수정 후 반드시 서버 재시작 필요
- 서버는 **반드시 프로젝트 루트**에서 실행해야 `.env` 파일을 찾음

### CLIP 모델 로드 실패 (SSL 에러)

기업 네트워크에서 SSL 인증서 문제가 발생할 수 있습니다.
위의 "모델 가중치 다운로드" 섹션을 참고하여 수동으로 다운로드하세요.

### 낮은 탐지 정확도

모든 탐지 모델이 활성화되어 있는지 확인:
- Sightengine API 키 설정 확인
- CLIP 설치 확인: `uv pip install git+https://github.com/openai/CLIP.git`
- Cross-EfficientViT 모델 가중치 다운로드

## 프로젝트 구조

```
src/
├── domain/           # 비즈니스 핵심 로직
├── application/      # 유스케이스
├── infrastructure/   # 외부 서비스 연동
│   ├── ai/          # 얼굴 인식, CLIP, EfficientViT
│   ├── external/    # OpenAI, Sightengine
│   └── persistence/ # Qdrant, Neo4j
├── interfaces/       # API 라우터
└── main.py          # 앱 진입점
```

## 개발 환경

### pre-commit 훅 설정

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

### 린팅 & 포맷팅

```bash
uv run ruff check src --fix
uv run ruff format src
uv run basedpyright src
```

### 테스트

```bash
uv run pytest --cov=src
```
