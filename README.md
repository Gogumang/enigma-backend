# Enigma

AI 기반 로맨스 스캠 종합 예방 플랫폼 백엔드 API 서버

> 딥페이크 탐지, 대화 분석, 사기 이력 조회, 프로필 검색, URL 검사, 면역 훈련, 신고 가이드까지 — 로맨스 스캠의 모든 공격 벡터를 차단하는 올인원 솔루션

## 주요 기능

### 1. 딥페이크 탐지

- **이미지 탐지**: GenD-PE + SigLIP + Sightengine 앙상블
- **비디오 탐지**: GenConViT + Sightengine 앙상블
- **AI 생성 이미지 탐지**: DALL-E, Midjourney, Stable Diffusion 등 감지
- 의심 영역 좌표(마커) 및 히트맵 시각화 제공
- 이미지 품질 기반 신뢰도 자동 보정

### 2. 대화 분석

- GPT-4o 기반 로맨스 스캠 8대 패턴 분석
- **Qdrant 벡터 검색** 기반 의미적 유사도 매칭 (RAG)
- **Neo4j 그래프 DB** 기반 관계/맥락 분석
- 대화 스크린샷 OCR 분석 지원
- 친구 대화 오탐 방지 (게임/캐주얼 맥락 감지)

### 3. 사기 이력 조회

- **경찰청 사이버범죄 API** 연동 — 전화번호/계좌번호/이메일 사기 이력 조회
- **사이버캅 API** 연동 — 다중 소스 교차 검증
- 전화번호/계좌번호 패턴 분석 (지역번호, 은행 코드 등)
- 위험도 판정 (SAFE / WARNING / DANGER)

### 4. 프로필 검색

- 얼굴 인식 기반 검색 (DeepFace VGG-Face)
- 역이미지 검색 링크 제공 (Google, Bing, TinEye, Yandex)
- 소셜 미디어 프로필 탐색 (Instagram, Facebook, LinkedIn, Twitter)
- 얼굴 감지 및 크롭 (MTCNN)

### 5. URL 검사

- **Google Safe Browsing API** 연동
- 단축 URL 자동 확장 (20+ 서비스 지원)
- 피싱/타이포스쿼팅 탐지 (Google, Facebook, Naver, Kakao 등)
- 위험 패턴 분석 (9개 카테고리)
- 위험도 점수화 (0-100)

### 6. 면역 훈련

- **LangGraph** 기반 AI 스캐머 시뮬레이션
- 다양한 페르소나 캐릭터 (난이도별)
- 5턴 대화 기반 훈련 + 스코어링
- 전술 탐지 및 피드백 제공

### 7. 신고 가이드

- GPT-4o 기반 신고서 초안 자동 생성
- 맞춤형 신고 절차 안내
- 긴급 조치 가이드 제공

### 8. 종합 분석

- 딥페이크 + 프로필 + 대화 + 사기조회 + URL 검사를 한 번에 실행
- 병렬 처리로 빠른 결과 반환

## 기술 스택

| 분류 | 기술 |
|------|------|
| **Framework** | FastAPI, Pydantic v2 |
| **Architecture** | DDD (Domain-Driven Design) |
| **LLM** | OpenAI GPT-4o |
| **AI Workflow** | LangGraph, LangChain |
| **Deepfake 탐지** | GenD-PE, GenConViT-ED, SigLIP-L/14 |
| **얼굴 인식** | DeepFace (VGG-Face), MediaPipe, MTCNN |
| **외부 API** | Sightengine, Google Safe Browsing, 경찰청 API, 사이버캅 API |
| **Vector DB** | Qdrant (대화 패턴 의미 검색) |
| **Graph DB** | Neo4j (관계 분석) |
| **이미지 처리** | OpenCV, Pillow, Albumentations |
| **HTTP** | httpx (비동기) |
| **배포** | Docker, Nginx, GitHub Actions |

## 탐지 모델

### 이미지 딥페이크 탐지 앙상블

**Primary (GenD-PE 사용 가능 시)**

| 모델 | 가중치 | 역할 |
|------|--------|------|
| GenD-PE (`yermandy/GenD_PE_L`) | 55% | 딥페이크 탐지 + 히트맵 + 6종 알고리즘 검사 |
| Sightengine (deepfake) | 25% | 얼굴 조작 탐지 |
| Sightengine (genai) | 20% | AI 생성 이미지 탐지 |

**Fallback (GenD-PE 미사용 시)**

| 모델 | 가중치 | 역할 |
|------|--------|------|
| SigLIP-L/14 (`google/siglip-large-patch16-384`) | 35% | Zero-shot AI 생성 이미지 탐지 |
| Sightengine (deepfake) | 35% | 얼굴 조작 탐지 |
| Sightengine (genai) | 30% | AI 생성 이미지 탐지 |

### 비디오 딥페이크 탐지 앙상블

| 모델 | 가중치 | 역할 |
|------|--------|------|
| GenConViT-ED (`Deressa/GenConViT`) | 70% | 비디오 딥페이크 (DFDC+FF+++CelebDF+TIMIT AUC 0.993) |
| Sightengine | 30% | 비디오 딥페이크 탐지 |

### 알고리즘 체크 항목

| 검사 | 설명 |
|------|------|
| Frequency Analysis | 주파수 영역 이상 탐지 |
| Skin Texture | 피부 텍스처 일관성 |
| Color Consistency | 색상 분포 일관성 |
| Edge Artifacts | 경계 부자연스러움 |
| Noise Pattern | 노이즈 패턴 불일치 |
| Compression Artifacts | 압축 아티팩트 이상 |

## 설치

### 1. 의존성 설치

```bash
# uv 사용 (권장)
uv sync

# 또는 pip 사용
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. 환경 변수 설정

`.env` 파일을 **프로젝트 루트**에 생성:

```bash
# .env
PORT=4000
ENV=local

# Sightengine (딥페이크 탐지) - 필수
SIGHTENGINE_API_USER=your_api_user
SIGHTENGINE_API_SECRET=your_api_secret

# OpenAI (대화 분석, 신고 가이드) - 필수
OPENAI_API_KEY=your_openai_key

# Qdrant (대화 패턴 벡터 검색)
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# Neo4j (관계 분석)
NEO4J_URI=neo4j://your_uri
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# 경찰청 사이버범죄 API (사기 조회)
POLICE_FRAUD_URL=your_url
POLICE_FRAUD_KEY=your_key

# 사이버캅 API (사기 조회)
CYBERCOP_URL=your_url

# Google Safe Browsing (URL 검사)
GOOGLE_SAFE_BROWSING_KEY=your_key
```

### 3. 모델 가중치 (자동 다운로드)

GenD-PE, SigLIP, GenConViT 모델 가중치는 서버 첫 실행 시 HuggingFace에서 자동 다운로드됩니다.

```bash
# 사전 다운로드 (선택)
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Deressa/GenConViT', 'genconvit_ed_inference.pth')"
```

## 실행 방법

### 방법 1: 가상환경 활성화 후 실행 (권장)

```bash
source .venv/bin/activate
python -m src.main
```

### 방법 2: uvicorn 직접 실행

```bash
source .venv/bin/activate
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
| **딥페이크** |
| POST | `/api/deepfake/analyze/image` | 이미지 딥페이크 분석 |
| POST | `/api/deepfake/analyze/video` | 영상 딥페이크 분석 |
| **대화 분석** |
| POST | `/api/chat/analyze` | 대화 메시지 분석 |
| POST | `/api/chat/analyze-screenshot` | 대화 스크린샷 분석 (OCR) |
| **사기 조회** |
| POST | `/api/fraud/check` | 전화번호/계좌/이메일 사기 이력 조회 |
| **프로필 검색** |
| POST | `/api/profile/detect-faces` | 이미지에서 얼굴 감지 |
| POST | `/api/profile/search` | 프로필 검색 (이미지/텍스트) |
| **URL 검사** |
| POST | `/api/url/check` | URL 안전성 검사 |
| **면역 훈련** |
| GET | `/api/training/personas` | 페르소나 목록 조회 |
| POST | `/api/training/start` | 훈련 세션 시작 |
| POST | `/api/training/message` | 훈련 메시지 전송 |
| POST | `/api/training/end` | 훈련 세션 종료 |
| **신고** |
| POST | `/api/report` | 신고 저장 |
| GET | `/api/report/{id}` | 신고 조회 |
| POST | `/api/report/check` | 신고 이력 확인 |
| POST | `/api/report/guide` | 신고 가이드 생성 |
| **통합 검증** |
| POST | `/api/verify/check` | 자동 감지 후 검증 (URL/전화/계좌) |
| POST | `/api/verify/url` | URL 검증 |
| POST | `/api/verify/phone` | 전화번호 검증 |
| POST | `/api/verify/account` | 계좌번호 검증 |
| **종합 분석** |
| POST | `/api/comprehensive/analyze` | 올인원 분석 |
| **시스템** |
| GET | `/api/health` | 헬스체크 |

## 사용 예시

### 이미지 딥페이크 분석

```bash
curl -X POST "http://localhost:4000/api/deepfake/analyze/image" \
  -F "file=@image.jpg"
```

### 대화 분석

```bash
curl -X POST "http://localhost:4000/api/chat/analyze" \
  -H "Content-Type: application/json" \
  -d '{"messages": "상대방: 나는 너를 정말 사랑해. 우리 만난 지 3일밖에 안 됐지만 운명이야.\n나: 감사합니다..."}'
```

### 사기 이력 조회

```bash
curl -X POST "http://localhost:4000/api/fraud/check" \
  -H "Content-Type: application/json" \
  -d '{"type": "PHONE", "value": "01012345678"}'
```

### URL 검사

```bash
curl -X POST "http://localhost:4000/api/url/check" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://suspicious-site.xyz/login"}'
```

## Docker 배포

```bash
# 빌드 및 실행
docker-compose -f docker-compose.prod.yml up -d --build

# 로그 확인
docker-compose -f docker-compose.prod.yml logs -f
```

## 프로젝트 구조

```
src/
├── domain/               # 비즈니스 핵심 로직 (엔티티, 값 객체)
│   ├── deepfake/         # 딥페이크 분석 도메인
│   ├── chat/             # 대화 분석 도메인
│   ├── fraud/            # 사기 조회 도메인
│   ├── profile/          # 프로필 검색 도메인
│   └── report/           # 신고 도메인
├── application/          # 유스케이스 (비즈니스 흐름)
│   ├── deepfake/         # 이미지/비디오 분석
│   ├── chat/             # 대화 분석
│   ├── fraud/            # 사기 이력 조회
│   ├── profile/          # 프로필 검색
│   ├── training/         # 면역 훈련 (LangGraph)
│   └── report/           # 신고 가이드 생성
├── infrastructure/       # 외부 서비스 연동
│   ├── ai/               # GenD-PE, GenConViT, SigLIP, DeepFace, MediaPipe
│   ├── external/         # Sightengine, OpenAI, 경찰청, 사이버캅
│   └── persistence/      # Qdrant, Neo4j
├── interfaces/api/       # FastAPI 라우터
└── main.py               # 앱 진입점
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
