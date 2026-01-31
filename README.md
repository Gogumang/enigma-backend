# Love Guard API

로맨스 스캠 예방을 위한 백엔드 API 서버

## 주요 기능

### 딥페이크 검사기
- 이미지/영상 딥페이크 탐지 (Sightengine API)
- GPT-4o Vision 기반 상세 분석
- 의심 영역 좌표 및 분석 이유 제공

### 대화 분석
- AI 기반 로맨스 스캠 패턴 분석
- 스크린샷 분석 지원
- **Qdrant 벡터 검색** 기반 의미적 유사도 매칭

### 스캐머 네트워크 분석
- **Neo4j 그래프 DB** 기반 스캐머 관계 추적
- 같은 계좌/전화번호/프로필 사진 사용 스캐머 연결
- 스캠 조직 네트워크 시각화
- 연관 스캐머 자동 탐지

### 프로필 검색
- 얼굴 인식 기반 스캐머 DB 검색 (DeepFace)
- 역이미지 검색 링크 제공

### URL 검사
- 피싱 URL 탐지

## 기술 스택

- **Framework**: FastAPI
- **Architecture**: DDD (Domain-Driven Design)
- **AI Services**: OpenAI GPT-4o, Sightengine
- **Face Recognition**: DeepFace (VGG-Face)
- **Vector DB**: Qdrant (대화 패턴 의미 검색)
- **Graph DB**: Neo4j (스캐머 네트워크 분석)

## 환경 변수

```bash
# .env
PORT=3001
CORS_ORIGIN=http://localhost:3000

# Sightengine (딥페이크 탐지)
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

## 실행 방법

### 1. 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성하고 위의 환경 변수를 설정하세요.

### 2. 설치 및 실행

#### uv 사용 (권장)

```bash
# 의존성 설치 및 가상환경 생성
uv sync

# 서버 실행
uv run python -m src.main

# 또는 uvicorn 직접 사용
uv run uvicorn src.main:app --reload --port 3001
```

#### pip 사용

```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 의존성 설치
pip install -e ".[dev]"

# 서버 실행
python -m src.main

# 또는 uvicorn 직접 사용
uvicorn src.main:app --reload --port 3001
```

### 3. 접속 확인

서버 실행 후 아래 주소로 접속:
- API 서버: http://localhost:3001
- API 문서 (Swagger): http://localhost:3001/api/docs

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/deepfake/analyze/image` | 이미지 딥페이크 분석 |
| POST | `/api/deepfake/analyze/video` | 영상 딥페이크 분석 |
| POST | `/api/chat/analyze` | 대화 분석 |
| POST | `/api/chat/analyze-screenshot` | 스크린샷 분석 |
| POST | `/api/network/report` | 스캐머 신고 |
| GET | `/api/network/analyze/{id}` | 스캐머 네트워크 분석 |
| GET | `/api/network/search/account/{number}` | 계좌번호로 스캐머 검색 |
| GET | `/api/network/search/phone/{number}` | 전화번호로 스캐머 검색 |
| GET | `/api/network/stats` | 네트워크 통계 |
| POST | `/api/profile/search` | 프로필 검색 |
| POST | `/api/profile/report` | 스캐머 신고 |
| POST | `/api/url/check` | URL 검사 |

## API 문서

http://localhost:3001/api/docs

## 데이터베이스 구조

### Qdrant (벡터 DB) - 대화 패턴
```
Collection: scam_patterns
- text: 스캠 문구
- category: love_bombing, financial_request, urgency, etc.
- severity: 위험도 (1-10)
- description: 설명
- examples: 예시 문구들
```

### Neo4j (그래프 DB) - 스캐머 네트워크
```
Nodes:
- Scammer: 스캐머 정보
- BankAccount: 사용된 계좌
- Phone: 사용된 전화번호
- ProfilePhoto: 프로필 사진 해시
- ScamPattern: 사용된 스캠 패턴

Relationships:
- (Scammer)-[:USED_ACCOUNT]->(BankAccount)
- (Scammer)-[:USED_PHONE]->(Phone)
- (Scammer)-[:USED_PHOTO]->(ProfilePhoto)
- (Scammer)-[:USED_PATTERN]->(ScamPattern)
```

## 프로젝트 구조

```
src/
├── domain/           # 비즈니스 핵심 로직
├── application/      # 유스케이스
├── infrastructure/   # 외부 서비스 연동
│   ├── ai/          # 얼굴 인식
│   ├── external/    # OpenAI, Sightengine
│   └── persistence/ # Qdrant, Neo4j
├── interfaces/       # API 라우터
└── main.py          # 앱 진입점
```

## 개발 환경

### pre-commit 훅 설정

```bash
# pre-commit 설치 (dev 의존성에 포함)
uv sync

# 훅 설치
uv run pre-commit install

# 수동 실행
uv run pre-commit run --all-files
```

### 린팅 & 포맷팅

```bash
# ruff 검사
uv run ruff check src

# ruff 자동 수정
uv run ruff check src --fix

# 코드 포맷팅
uv run ruff format src

# 타입 체크
uv run basedpyright src
```

### 테스트

```bash
# 테스트 실행
uv run pytest

# 커버리지 포함
uv run pytest --cov=src
```
