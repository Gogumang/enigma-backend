"""
Qdrant 기반 스캠 패턴 벡터 DB 리포지토리
- 의미적 유사도 검색으로 스캠 패턴 매칭
- OpenAI 임베딩 사용
"""
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.shared.config import get_settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "scam_patterns"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


@dataclass
class ScamPattern:
    """스캠 패턴 (위험 패턴 + 안전 패턴)"""
    id: str
    text: str
    category: str  # love_bombing, financial_request, friend_gaming, friend_betting, etc.
    severity: int  # 1-10 (안전 패턴은 음수 사용: -10 ~ -1)
    description: str
    examples: list[str]
    is_safe: bool = False  # True면 정상/안전 패턴


@dataclass
class RAGResult:
    """RAG 조회 결과"""
    matched_patterns: list[dict]
    safe_patterns: list[dict]  # 매칭된 안전 패턴
    risk_score: int
    risk_indicators: list[str]
    protective_indicators: list[str]  # 보호 지표


class QdrantScamRepository:
    """Qdrant 기반 스캠 패턴 리포지토리"""

    def __init__(self):
        settings = get_settings()
        self.qdrant_url = settings.qdrant_url
        self.qdrant_api_key = settings.qdrant_api_key
        self.openai_api_key = settings.openai_api_key
        self._qdrant: Optional[QdrantClient] = None
        self._openai: Optional[OpenAI] = None

    async def initialize(self) -> None:
        """Qdrant 연결 및 컬렉션 초기화"""
        if not self.qdrant_url or not self.qdrant_api_key:
            logger.warning("Qdrant not configured")
            return

        try:
            self._qdrant = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )
            self._openai = OpenAI(api_key=self.openai_api_key)

            # 컬렉션 생성 (없으면)
            collections = self._qdrant.get_collections().collections
            if not any(c.name == COLLECTION_NAME for c in collections):
                self._qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created collection: {COLLECTION_NAME}")

                # 초기 데이터 추가
                await self._seed_initial_data()

            # 기존 데이터 확인
            count = self._qdrant.count(collection_name=COLLECTION_NAME).count
            logger.info(f"QdrantScamRepository initialized with {count} patterns")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            self._qdrant = None

    def _get_embedding(self, text: str) -> list[float]:
        """텍스트 임베딩 생성"""
        if not self._openai:
            return [0.0] * EMBEDDING_DIM

        response = self._openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    async def _seed_initial_data(self) -> None:
        """초기 스캠 패턴 데이터 추가"""
        initial_patterns = [
            # Love Bombing (과도한 애정 표현)
            ScamPattern(
                id=str(uuid.uuid4()),
                text="처음 보는데 사랑에 빠졌어요",
                category="love_bombing",
                severity=7,
                description="첫 만남부터 강한 감정 표현은 로맨스 스캠의 전형적인 시작",
                examples=["운명인 것 같아", "첫눈에 반했어", "당신은 특별해"]
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="매일 보고싶어 미치겠어",
                category="love_bombing",
                severity=6,
                description="과도한 그리움 표현으로 감정적 의존 유도",
                examples=["하루도 못 살겠어", "당신 없이 못 살아", "매 순간 생각나"]
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="우리 빨리 결혼하자",
                category="love_bombing",
                severity=8,
                description="만난 지 얼마 안 되어 결혼 언급은 위험 신호",
                examples=["평생 함께하고 싶어", "내 아내가 되어줘", "같이 살자"]
            ),

            # Financial Request (금전 요청)
            ScamPattern(
                id=str(uuid.uuid4()),
                text="급하게 돈이 필요해",
                category="financial_request",
                severity=10,
                description="금전 요청은 로맨스 스캠의 핵심 목적",
                examples=["돈 좀 빌려줘", "송금해줄 수 있어?", "계좌로 보내줘"]
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="투자하면 큰 수익이 나",
                category="financial_request",
                severity=10,
                description="투자 권유는 로맨스 스캠과 투자 사기의 결합",
                examples=["코인 투자해봐", "같이 돈 벌자", "수익률 보장해"]
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="병원비가 부족해서",
                category="financial_request",
                severity=9,
                description="동정심을 이용한 금전 요청",
                examples=["수술비가 필요해", "치료비 좀 도와줘", "약값이 없어"]
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="세금 내야 돈을 찾을 수 있어",
                category="financial_request",
                severity=10,
                description="해외 송금 사기의 전형적인 수법",
                examples=["관세를 내야 해", "수수료만 보내줘", "서류비용이 필요해"]
            ),

            # Urgency (급박함)
            ScamPattern(
                id=str(uuid.uuid4()),
                text="지금 당장 보내줘야 해",
                category="urgency",
                severity=8,
                description="급박함을 강조해 판단력을 흐리게 함",
                examples=["오늘 안에 해야 해", "시간이 없어", "빨리 결정해"]
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="이번 기회 놓치면 안 돼",
                category="urgency",
                severity=7,
                description="FOMO(놓칠까 봐 두려움)를 자극",
                examples=["마지막 기회야", "한정된 자리야", "지금 아니면 안 돼"]
            ),

            # Isolation (고립 유도)
            ScamPattern(
                id=str(uuid.uuid4()),
                text="우리 둘만의 비밀이야",
                category="isolation",
                severity=9,
                description="주변인과의 상담을 막아 고립시킴",
                examples=["아무에게도 말하지 마", "부모님한테 비밀로 해", "친구들이 질투할 거야"]
            ),

            # Avoidance (회피)
            ScamPattern(
                id=str(uuid.uuid4()),
                text="영상통화는 지금 안 돼",
                category="avoidance",
                severity=8,
                description="실제 얼굴 확인을 피하는 것은 가짜 신원의 증거",
                examples=["카메라가 고장났어", "인터넷이 안 좋아", "나중에 하자"]
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="만나는 건 아직 이르지 않아?",
                category="avoidance",
                severity=7,
                description="직접 만남을 계속 미루는 패턴",
                examples=["출장 중이야", "해외에 있어서", "상황이 안 돼"]
            ),

            # Sob Story (불쌍한 사연)
            ScamPattern(
                id=str(uuid.uuid4()),
                text="사고가 나서 큰일이야",
                category="sob_story",
                severity=8,
                description="동정심을 유발하는 사연으로 금전 요청 준비",
                examples=["교통사고 났어", "도둑맞았어", "지갑을 잃어버렸어"]
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="가족이 아파서 힘들어",
                category="sob_story",
                severity=7,
                description="가족 질병을 이유로 동정심 유발",
                examples=["엄마가 암이야", "아이가 아파", "아버지 수술해야 해"]
            ),

            # Identity (신원 관련)
            ScamPattern(
                id=str(uuid.uuid4()),
                text="나는 군인이야 해외 파병 중",
                category="identity",
                severity=8,
                description="군인 신분 사칭은 흔한 로맨스 스캠 수법",
                examples=["의사야", "사업가야", "엔지니어야"]
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="곧 한국에 갈 거야",
                category="identity",
                severity=6,
                description="만남 약속을 반복하지만 실현하지 않음",
                examples=["다음 달에 가", "휴가 나오면 만나", "곧 전역해"]
            ),

            # ==================== 정상/안전 패턴 (친구 대화) ====================

            # 게임 내기 패턴 (안전)
            ScamPattern(
                id=str(uuid.uuid4()),
                text="야 롤 한판 ㄱ?",
                category="friend_gaming",
                severity=-8,
                description="친구 간 게임 제안 - 정상적인 대화",
                examples=["롤 ㄱ?", "한판 할래?", "듀오 ㄱㄱ", "랭크 돌리자"],
                is_safe=True
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="졌으니까 만원 내놔",
                category="friend_betting",
                severity=-9,
                description="친구 간 게임 내기 결과 - 정상적인 대화",
                examples=["내기 졌으니까 내놔", "만원빵", "빵 내놔", "한턱 쏴"],
                is_safe=True
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="이겼으니까 계좌번호 보내",
                category="friend_betting",
                severity=-9,
                description="친구 간 게임 내기 후 송금 요청 - 정상적인 대화",
                examples=["계좌 보내", "계좌번호 알려줘", "카카오페이로 보내"],
                is_safe=True
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="넌 뒤졌다 ㅋㅋㅋ",
                category="friend_casual",
                severity=-7,
                description="친구 간 장난 - 정상적인 대화",
                examples=["죽었어", "각오해", "두고봐", "나한테 안돼"],
                is_safe=True
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="치킨 사줘",
                category="friend_casual",
                severity=-6,
                description="친구 간 밥/음식 사달라는 요청 - 정상적인 대화",
                examples=["밥 사줘", "커피 사줘", "한턱내", "쏴라"],
                is_safe=True
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="심심한데 뭐해",
                category="friend_casual",
                severity=-5,
                description="친구 간 안부 인사 - 정상적인 대화",
                examples=["뭐해?", "심심해", "놀자", "시간 있어?"],
                is_safe=True
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="ㅋㅋㅋㅋ 진짜?",
                category="friend_casual",
                severity=-5,
                description="친구 간 반응 - 정상적인 대화",
                examples=["ㅎㅎㅎ", "ㄹㅇ?", "ㄴㄴ", "ㅇㅇ", "ㅇㅋ"],
                is_safe=True
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="택시비 좀 빌려줘 나중에 갚을게",
                category="friend_money",
                severity=-4,
                description="친구 간 소액 빌리기 - 맥락에 따라 정상",
                examples=["만원만 빌려줘", "점심값 좀", "교통비 좀"],
                is_safe=True
            ),
            ScamPattern(
                id=str(uuid.uuid4()),
                text="오늘 저녁에 만나자",
                category="friend_meetup",
                severity=-6,
                description="친구 간 만남 약속 - 정상적인 대화",
                examples=["언제 볼래?", "주말에 만나자", "오랜만에 보자"],
                is_safe=True
            ),
        ]

        points = []
        for pattern in initial_patterns:
            # 패턴 텍스트와 예시를 합쳐서 임베딩
            full_text = f"{pattern.text} {' '.join(pattern.examples)}"
            embedding = self._get_embedding(full_text)

            points.append(PointStruct(
                id=pattern.id,
                vector=embedding,
                payload={
                    "text": pattern.text,
                    "category": pattern.category,
                    "severity": pattern.severity,
                    "description": pattern.description,
                    "examples": pattern.examples,
                    "is_safe": pattern.is_safe,
                }
            ))

        if points and self._qdrant:
            self._qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )
            logger.info(f"Seeded {len(points)} initial scam patterns")

    def is_connected(self) -> bool:
        return self._qdrant is not None

    async def search_similar(self, text: str, limit: int = 10) -> RAGResult:
        """텍스트와 유사한 패턴 검색 (위험 + 안전 패턴)"""
        if not self._qdrant or not self._openai:
            return RAGResult(
                matched_patterns=[],
                safe_patterns=[],
                risk_score=0,
                risk_indicators=[],
                protective_indicators=[]
            )

        try:
            # 입력 텍스트 임베딩
            query_embedding = self._get_embedding(text)

            # 유사도 검색 (더 많이 가져와서 안전/위험 패턴 모두 확인)
            response = self._qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=limit,
                score_threshold=0.25,  # 최소 유사도 25%
            )
            results = response.points

            matched_patterns = []  # 위험 패턴
            safe_patterns = []     # 안전 패턴
            total_severity = 0
            safe_score = 0
            risk_indicators = []
            protective_indicators = []
            categories_found = set()
            safe_categories_found = set()

            for result in results:
                payload = result.payload
                similarity = result.score
                is_safe = payload.get("is_safe", False)

                pattern_info = {
                    "text": payload["text"],
                    "category": payload["category"],
                    "severity": payload["severity"],
                    "description": payload["description"],
                    "similarity": round(similarity * 100, 1),
                    "is_safe": is_safe,
                }

                if is_safe:
                    # 안전 패턴
                    safe_patterns.append(pattern_info)

                    # 안전 점수 계산 (음수 severity의 절댓값 사용)
                    weighted_safe = abs(payload["severity"]) * similarity
                    safe_score += weighted_safe
                    safe_categories_found.add(payload["category"])

                    # 보호 지표 생성
                    if similarity >= 0.6:
                        protective_indicators.append(
                            f"'{payload['text']}' - {payload['description']} ({similarity*100:.0f}% 일치)"
                        )
                    elif similarity >= 0.4:
                        protective_indicators.append(
                            f"친구 대화 패턴 감지: {payload['category']} ({similarity*100:.0f}%)"
                        )
                else:
                    # 위험 패턴
                    matched_patterns.append(pattern_info)

                    # 위험도 계산
                    weighted_severity = payload["severity"] * similarity
                    total_severity += weighted_severity
                    categories_found.add(payload["category"])

                    # 위험 지표 생성
                    if similarity >= 0.7:
                        risk_indicators.append(
                            f"'{payload['text']}' 패턴과 {similarity*100:.0f}% 유사 - {payload['description']}"
                        )
                    elif similarity >= 0.5:
                        risk_indicators.append(
                            f"'{payload['text']}' 패턴 감지 ({similarity*100:.0f}% 일치)"
                        )

            # 최종 위험 점수 계산
            # 기본 위험 점수
            base_risk_score = min(100, int(total_severity * 10))

            # 안전 패턴이 감지되면 위험도 감소
            if safe_score > 0:
                # 안전 점수가 높을수록 위험도 감소
                safety_factor = min(0.8, safe_score * 0.1)  # 최대 80% 감소
                base_risk_score = int(base_risk_score * (1 - safety_factor))

                if safe_categories_found:
                    protective_indicators.append(
                        f"친구 대화 맥락 감지: {', '.join(safe_categories_found)}"
                    )

            # 여러 위험 카테고리가 감지되면 추가 위험 (안전 패턴이 없을 때만)
            if len(categories_found) >= 3 and safe_score == 0:
                base_risk_score = min(100, base_risk_score + 15)
                risk_indicators.append(f"다중 스캠 패턴 감지: {', '.join(categories_found)}")

            # 안전 패턴이 위험 패턴보다 유사도가 높으면 위험도 더 낮춤
            if safe_patterns and matched_patterns:
                max_safe_sim = max(p["similarity"] for p in safe_patterns)
                max_risk_sim = max(p["similarity"] for p in matched_patterns)
                if max_safe_sim > max_risk_sim:
                    base_risk_score = int(base_risk_score * 0.5)
                    protective_indicators.append(
                        f"안전 패턴({max_safe_sim:.0f}%)이 위험 패턴({max_risk_sim:.0f}%)보다 더 유사"
                    )

            return RAGResult(
                matched_patterns=matched_patterns,
                safe_patterns=safe_patterns,
                risk_score=max(0, base_risk_score),
                risk_indicators=risk_indicators,
                protective_indicators=protective_indicators
            )

        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return RAGResult(
                matched_patterns=[],
                safe_patterns=[],
                risk_score=0,
                risk_indicators=[],
                protective_indicators=[]
            )

    async def add_pattern(self, pattern: ScamPattern) -> bool:
        """새 패턴 추가 (위험/안전 패턴 모두 가능)"""
        if not self._qdrant or not self._openai:
            return False

        try:
            full_text = f"{pattern.text} {' '.join(pattern.examples)}"
            embedding = self._get_embedding(full_text)

            self._qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=[PointStruct(
                    id=pattern.id,
                    vector=embedding,
                    payload={
                        "text": pattern.text,
                        "category": pattern.category,
                        "severity": pattern.severity,
                        "description": pattern.description,
                        "examples": pattern.examples,
                        "is_safe": pattern.is_safe,
                    }
                )]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add pattern: {e}")
            return False

    async def get_stats(self) -> dict:
        """통계 조회"""
        if not self._qdrant:
            return {"total_patterns": 0}

        try:
            count = self._qdrant.count(collection_name=COLLECTION_NAME).count
            return {"total_patterns": count}
        except Exception:
            return {"total_patterns": 0}

    async def close(self) -> None:
        """연결 종료"""
        if self._qdrant:
            self._qdrant.close()
