"""
Neo4j 기반 사용자 관계 및 대화 컨텍스트 리포지토리
- 사용자 간 관계(친구, 연인, 모르는 사람 등) 관리
- 대화 이력 및 맥락 저장
- 관계 기반 신뢰도 계산
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from neo4j import AsyncGraphDatabase

from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class RelationshipType(str, Enum):
    """관계 유형"""
    FRIEND = "friend"           # 친구
    CLOSE_FRIEND = "close_friend"  # 절친
    FAMILY = "family"           # 가족
    LOVER = "lover"             # 연인
    ACQUAINTANCE = "acquaintance"  # 지인
    ONLINE_FRIEND = "online_friend"  # 온라인 친구
    MATCHED = "matched"         # 매칭앱에서 만남
    STRANGER = "stranger"       # 모르는 사람
    UNKNOWN = "unknown"         # 관계 불명


@dataclass
class UserRelationship:
    """사용자 간 관계 정보"""
    user_id: str
    other_user_id: str
    relationship_type: RelationshipType
    trust_level: float  # 0.0 ~ 1.0
    known_since: Optional[datetime] = None
    interaction_count: int = 0
    financial_request_count: int = 0  # 금전 요청 횟수
    platform: Optional[str] = None  # 만난 플랫폼 (tinder, bumble, etc.)


@dataclass
class ConversationContext:
    """대화 맥락 정보"""
    context_type: str  # gaming, work, dating, etc.
    keywords: list[str]
    confidence: float


@dataclass
class RelationshipAnalysisResult:
    """관계 기반 분석 결과"""
    relationship: Optional[UserRelationship]
    trust_modifier: float  # 최종 신뢰도 수정자 (0.0 ~ 1.0)
    context: Optional[ConversationContext]
    risk_factors: list[str]
    protective_factors: list[str]


# 관계 유형별 기본 신뢰도
DEFAULT_TRUST_LEVELS = {
    RelationshipType.FAMILY: 0.95,
    RelationshipType.CLOSE_FRIEND: 0.9,
    RelationshipType.LOVER: 0.85,
    RelationshipType.FRIEND: 0.8,
    RelationshipType.ACQUAINTANCE: 0.5,
    RelationshipType.ONLINE_FRIEND: 0.4,
    RelationshipType.MATCHED: 0.2,
    RelationshipType.STRANGER: 0.1,
    RelationshipType.UNKNOWN: 0.3,
}

# 맥락별 위험도 수정자 (낮을수록 위험도 낮춤 = 안전)
CONTEXT_MODIFIERS = {
    "gaming_betting": 0.1,   # 게임 내기 → 위험도 대폭 낮춤 (친구 대화)
    "friend_casual": 0.15,   # 친구 캐주얼 대화 → 위험도 대폭 낮춤
    "gaming": 0.2,           # 일반 게임 관련 → 위험도 낮춤
    "work": 0.4,             # 업무 관련
    "casual": 0.3,           # 일상 대화
    "dating": 0.7,           # 데이팅 → 주의 필요
    "financial": 0.8,        # 일반 금융 관련 → 주의 필요
    "financial_scam": 1.0,   # 스캠 의심 금융 → 최고 주의
    "emergency": 0.9,        # 긴급 상황 → 높은 주의
}


class Neo4jRelationshipRepository:
    """Neo4j 기반 관계 리포지토리"""

    def __init__(self):
        settings = get_settings()
        self.uri = settings.neo4j_uri
        self.username = settings.neo4j_username
        self.password = settings.neo4j_password
        self._driver = None

    async def initialize(self) -> None:
        """Neo4j 연결 초기화"""
        if not self.uri:
            logger.warning("Neo4j URI not configured for relationship repository")
            return

        try:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            async with self._driver.session() as session:
                await session.run("RETURN 1")
            logger.info("Neo4jRelationshipRepository connected")

            await self._create_indexes()
            await self._create_constraints()

        except Exception as e:
            logger.error(f"Failed to initialize Neo4jRelationshipRepository: {e}")
            self._driver = None

    async def _create_indexes(self) -> None:
        """인덱스 생성"""
        if not self._driver:
            return

        async with self._driver.session() as session:
            # User 노드 인덱스
            await session.run(
                "CREATE INDEX user_id IF NOT EXISTS FOR (u:User) ON (u.id)"
            )
            # Conversation 노드 인덱스
            await session.run(
                "CREATE INDEX conversation_id IF NOT EXISTS FOR (c:Conversation) ON (c.id)"
            )

    async def _create_constraints(self) -> None:
        """제약조건 생성"""
        if not self._driver:
            return

        async with self._driver.session() as session:
            try:
                await session.run(
                    "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE"
                )
            except Exception:
                pass  # 이미 존재하면 무시

    async def close(self) -> None:
        """연결 종료"""
        if self._driver:
            await self._driver.close()

    def is_connected(self) -> bool:
        return self._driver is not None

    # ==================== 사용자 관리 ====================

    async def create_user(self, user_id: str, name: Optional[str] = None) -> bool:
        """사용자 생성"""
        if not self._driver:
            return False

        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (u:User {id: $user_id})
                SET u.name = $name,
                    u.created_at = coalesce(u.created_at, datetime()),
                    u.updated_at = datetime()
                """,
                user_id=user_id,
                name=name
            )
        return True

    # ==================== 관계 관리 ====================

    async def set_relationship(
        self,
        user_id: str,
        other_user_id: str,
        relationship_type: RelationshipType,
        trust_level: Optional[float] = None,
        platform: Optional[str] = None
    ) -> bool:
        """두 사용자 간 관계 설정"""
        if not self._driver:
            return False

        # 신뢰도가 지정되지 않으면 기본값 사용
        if trust_level is None:
            trust_level = DEFAULT_TRUST_LEVELS.get(relationship_type, 0.3)

        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (u1:User {id: $user_id})
                MERGE (u2:User {id: $other_user_id})
                MERGE (u1)-[r:KNOWS]->(u2)
                SET r.type = $rel_type,
                    r.trust_level = $trust_level,
                    r.platform = $platform,
                    r.known_since = coalesce(r.known_since, datetime()),
                    r.interaction_count = coalesce(r.interaction_count, 0),
                    r.financial_request_count = coalesce(r.financial_request_count, 0),
                    r.updated_at = datetime()
                """,
                user_id=user_id,
                other_user_id=other_user_id,
                rel_type=relationship_type.value,
                trust_level=trust_level,
                platform=platform
            )
        return True

    async def get_relationship(
        self,
        user_id: str,
        other_user_id: str
    ) -> Optional[UserRelationship]:
        """두 사용자 간 관계 조회"""
        if not self._driver:
            return None

        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (u1:User {id: $user_id})-[r:KNOWS]->(u2:User {id: $other_user_id})
                RETURN r.type as type, r.trust_level as trust_level,
                       r.known_since as known_since, r.interaction_count as interaction_count,
                       r.financial_request_count as financial_request_count, r.platform as platform
                """,
                user_id=user_id,
                other_user_id=other_user_id
            )
            record = await result.single()

            if record:
                return UserRelationship(
                    user_id=user_id,
                    other_user_id=other_user_id,
                    relationship_type=RelationshipType(record["type"]),
                    trust_level=record["trust_level"],
                    known_since=record["known_since"].to_native() if record["known_since"] else None,
                    interaction_count=record["interaction_count"] or 0,
                    financial_request_count=record["financial_request_count"] or 0,
                    platform=record["platform"]
                )

        return None

    async def increment_interaction(
        self,
        user_id: str,
        other_user_id: str,
        is_financial_request: bool = False
    ) -> bool:
        """상호작용 카운트 증가"""
        if not self._driver:
            return False

        async with self._driver.session() as session:
            if is_financial_request:
                await session.run(
                    """
                    MATCH (u1:User {id: $user_id})-[r:KNOWS]->(u2:User {id: $other_user_id})
                    SET r.interaction_count = coalesce(r.interaction_count, 0) + 1,
                        r.financial_request_count = coalesce(r.financial_request_count, 0) + 1,
                        r.updated_at = datetime()
                    """,
                    user_id=user_id,
                    other_user_id=other_user_id
                )
            else:
                await session.run(
                    """
                    MATCH (u1:User {id: $user_id})-[r:KNOWS]->(u2:User {id: $other_user_id})
                    SET r.interaction_count = coalesce(r.interaction_count, 0) + 1,
                        r.updated_at = datetime()
                    """,
                    user_id=user_id,
                    other_user_id=other_user_id
                )
        return True

    # ==================== 대화 맥락 분석 ====================

    def detect_context(self, message: str) -> ConversationContext:
        """메시지에서 대화 맥락 감지 (강화된 패턴)"""
        message_lower = message.lower()

        # 맥락별 키워드 (우선순위 순서로 정렬)
        context_keywords = {
            # 게임/내기 - 가장 높은 우선순위 (친구 대화 신호)
            "gaming_betting": [
                "롤", "lol", "배그", "pubg", "오버워치", "발로란트", "게임",
                "한판", "듀오", "솔랭", "랭크", "피드", "캐리",
                "빵", "내기", "만원빵", "만원 빵", "오천빵", "천원빵",
                "졌", "이겼", "내놔", "한턱", "빵셔틀",
                "ㄱ?", "ㄱㄱ", "ㄴㄴ", "고고", "ㅇㅋ"
            ],
            # 친구 캐주얼 대화
            "friend_casual": [
                "ㅋㅋ", "ㅎㅎ", "ㅋㅋㅋ", "ㅎㅎㅎ", "ㅋㅋㅋㅋ",
                "ㄴㄴ", "ㅇㅇ", "ㅇㅋ", "ㄱㅊ", "ㅈㅅ",
                "뭐해", "심심", "야", "님", "ㅅㅂ", "ㅁㅊ",
                "뒤졌", "죽었", "각오해", "두고봐",
                "사줘", "밥", "커피", "술", "치킨"
            ],
            # 일반 게임 (내기 없이)
            "gaming": [
                "스팀", "steam", "닌텐도", "플스", "엑박", "xbox",
                "마크", "마인크래프트", "리그오브레전드", "스타크래프트",
                "스킨", "아이템", "뽑기", "가챠", "과금"
            ],
            # 업무
            "work": [
                "회사", "업무", "미팅", "프로젝트", "출장", "야근",
                "월급", "보고서", "상사", "부장님", "팀장님", "회의"
            ],
            # 로맨스 스캠 의심 - 금융
            "financial_scam": [
                "투자 기회", "원금 보장", "고수익", "비트코인 투자",
                "코인 투자", "수익률", "배당", "리딩방"
            ],
            # 일반 금융 (친구 간 송금은 gaming_betting에서 처리)
            "financial": [
                "투자", "코인", "주식", "수익", "원금", "이자", "대출"
            ],
            # 긴급 상황 (스캠 의심)
            "emergency": [
                "급해", "긴급", "사고가 났", "병원비", "수술비",
                "아파서", "도와줘", "지금 당장"
            ],
            # 데이팅/로맨스
            "dating": [
                "데이트", "사랑해", "보고싶", "연애", "썸",
                "운명", "소울메이트", "첫눈에"
            ],
            # 캐주얼 (기본)
            "casual": [
                "뭐해", "심심", "안녕"
            ],
        }

        # 우선순위: gaming_betting > friend_casual > 나머지
        priority_order = [
            "gaming_betting", "friend_casual", "gaming",
            "work", "financial_scam", "financial",
            "emergency", "dating", "casual"
        ]

        detected_contexts = []
        for context_type in priority_order:
            keywords = context_keywords.get(context_type, [])
            matched = [kw for kw in keywords if kw in message_lower]
            if matched:
                # 매칭된 키워드 수에 따른 신뢰도
                confidence = min(1.0, len(matched) * 0.25)
                # 특정 맥락은 가중치 부여
                if context_type in ["gaming_betting", "friend_casual"]:
                    confidence = min(1.0, confidence * 1.5)  # 친구 대화 맥락 가중치
                detected_contexts.append((context_type, matched, confidence))

        if detected_contexts:
            # 우선순위와 confidence 고려
            # gaming_betting이나 friend_casual이 감지되면 최우선
            for ctx in detected_contexts:
                if ctx[0] in ["gaming_betting", "friend_casual"] and ctx[2] >= 0.3:
                    return ConversationContext(
                        context_type=ctx[0],
                        keywords=ctx[1],
                        confidence=ctx[2]
                    )

            # 그 외는 가장 높은 confidence 선택
            detected_contexts.sort(key=lambda x: x[2], reverse=True)
            best = detected_contexts[0]
            return ConversationContext(
                context_type=best[0],
                keywords=best[1],
                confidence=best[2]
            )

        return ConversationContext(
            context_type="casual",
            keywords=[],
            confidence=0.3
        )

    # ==================== 종합 분석 ====================

    async def analyze_relationship_context(
        self,
        message: str,
        sender_id: Optional[str] = None,
        receiver_id: Optional[str] = None
    ) -> RelationshipAnalysisResult:
        """관계 및 맥락 기반 종합 분석"""
        risk_factors = []
        protective_factors = []

        # 1. 관계 조회
        relationship = None
        if sender_id and receiver_id:
            relationship = await self.get_relationship(receiver_id, sender_id)

        # 2. 대화 맥락 감지
        context = self.detect_context(message)

        # 3. 신뢰도 계산
        if relationship:
            base_trust = relationship.trust_level

            # 금전 요청 이력이 많으면 신뢰도 하락
            if relationship.financial_request_count > 3:
                base_trust *= 0.7
                risk_factors.append(f"이전 금전 요청 {relationship.financial_request_count}회")

            # 상호작용이 많으면 신뢰도 약간 상승
            if relationship.interaction_count > 50:
                base_trust = min(1.0, base_trust * 1.1)
                protective_factors.append(f"활발한 대화 이력 ({relationship.interaction_count}회)")

            # 매칭앱에서 만난 경우 주의
            if relationship.platform in ["tinder", "bumble", "hinge"]:
                base_trust *= 0.8
                risk_factors.append(f"매칭앱({relationship.platform})에서 만남")

            # 관계 유형별 보호 요소
            if relationship.relationship_type in [RelationshipType.FAMILY, RelationshipType.CLOSE_FRIEND]:
                protective_factors.append(f"신뢰할 수 있는 관계: {relationship.relationship_type.value}")

        else:
            # 관계 정보 없음 → 기본 신뢰도
            base_trust = DEFAULT_TRUST_LEVELS[RelationshipType.UNKNOWN]
            risk_factors.append("관계 정보 없음")

        # 4. 맥락 기반 수정
        context_modifier = CONTEXT_MODIFIERS.get(context.context_type, 0.5)

        # 맥락별 보호/위험 요소 추가
        if context.context_type == "gaming_betting":
            protective_factors.append(f"게임 내기 맥락 감지 (키워드: {', '.join(context.keywords)})")
            protective_factors.append("친구 간 일상적인 게임 내기로 판단됨")
            # 게임 내기 맥락에서는 신뢰도 대폭 상승
            base_trust = min(1.0, base_trust + 0.4)
        elif context.context_type == "friend_casual":
            protective_factors.append(f"친구 캐주얼 대화 감지 (키워드: {', '.join(context.keywords)})")
            protective_factors.append("친구/지인 간 일상 대화로 판단됨")
            base_trust = min(1.0, base_trust + 0.3)
        elif context.context_type == "gaming":
            protective_factors.append(f"게임 관련 대화 (키워드: {', '.join(context.keywords)})")
        elif context.context_type == "financial_scam":
            risk_factors.append("스캠 의심 금융 패턴 감지")
            risk_factors.append(f"위험 키워드: {', '.join(context.keywords)}")
        elif context.context_type == "financial":
            risk_factors.append("금융/투자 관련 대화")
        elif context.context_type == "emergency":
            risk_factors.append("긴급 상황 언급")
        elif context.context_type == "dating":
            risk_factors.append("로맨스/데이팅 맥락 - 주의 필요")

        # 5. 최종 신뢰도 수정자 계산
        # trust_modifier가 높을수록 위험도가 낮아짐
        # context_modifier가 낮을수록 (친구 대화) 신뢰도가 높아짐
        trust_modifier = base_trust * (1 - context_modifier * 0.5)

        return RelationshipAnalysisResult(
            relationship=relationship,
            trust_modifier=trust_modifier,
            context=context,
            risk_factors=risk_factors,
            protective_factors=protective_factors
        )
