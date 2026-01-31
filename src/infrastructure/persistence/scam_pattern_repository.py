"""
Neo4j 기반 스캠 패턴 RAG 리포지토리
- ScamPhrase: 스캠에서 사용되는 문구
- ScamCase: 실제 스캠 사례
- ChatAnalysisLog: 분석 이력
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from neo4j import AsyncGraphDatabase

from src.shared.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ScamPhrase:
    """스캠 문구"""
    id: str
    text: str
    category: str  # love_bombing, financial_request, urgency, etc.
    severity: int  # 1-10
    usage_count: int
    examples: list[str]


@dataclass
class ScamCase:
    """스캠 사례"""
    id: str
    title: str
    description: str
    phrases_used: list[str]
    damage_amount: Optional[int]  # 피해액 (원)
    reported_at: datetime
    platform: str  # kakao, instagram, telegram, etc.


@dataclass
class RAGResult:
    """RAG 조회 결과"""
    matched_phrases: list[ScamPhrase]
    similar_cases: list[ScamCase]
    total_reports: int
    risk_indicators: list[str]


class ScamPatternRepository:
    """스캠 패턴 RAG 리포지토리"""

    def __init__(self):
        settings = get_settings()
        self.uri = settings.neo4j_uri
        self.username = settings.neo4j_username
        self.password = settings.neo4j_password
        self._driver = None

    async def initialize(self) -> None:
        """Neo4j 연결 및 스키마 초기화"""
        if not self.uri:
            logger.warning("Neo4j URI not configured")
            return

        try:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            async with self._driver.session() as session:
                await session.run("RETURN 1")
            logger.info("ScamPatternRepository connected to Neo4j")

            await self._create_schema()

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self._driver = None

    async def _create_schema(self) -> None:
        """스키마 및 인덱스 생성"""
        if not self._driver:
            return

        async with self._driver.session() as session:
            # 인덱스 생성
            await session.run(
                "CREATE INDEX scam_phrase_id IF NOT EXISTS FOR (p:ScamPhrase) ON (p.id)"
            )
            await session.run(
                "CREATE INDEX scam_phrase_category IF NOT EXISTS FOR (p:ScamPhrase) ON (p.category)"
            )
            await session.run(
                "CREATE INDEX scam_case_id IF NOT EXISTS FOR (c:ScamCase) ON (c.id)"
            )
            await session.run(
                "CREATE FULLTEXT INDEX scam_phrase_text IF NOT EXISTS FOR (p:ScamPhrase) ON EACH [p.text]"
            )
            logger.info("ScamPattern schema created")

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    def is_connected(self) -> bool:
        return self._driver is not None

    # ==================== 문구 관리 ====================

    async def save_phrase(self, phrase: ScamPhrase) -> None:
        """스캠 문구 저장"""
        if not self._driver:
            return

        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (p:ScamPhrase {id: $id})
                SET p.text = $text,
                    p.category = $category,
                    p.severity = $severity,
                    p.usage_count = $usage_count,
                    p.examples = $examples
                """,
                id=phrase.id,
                text=phrase.text,
                category=phrase.category,
                severity=phrase.severity,
                usage_count=phrase.usage_count,
                examples=phrase.examples
            )

    async def save_case(self, case: ScamCase) -> None:
        """스캠 사례 저장"""
        if not self._driver:
            return

        async with self._driver.session() as session:
            # 사례 저장
            await session.run(
                """
                MERGE (c:ScamCase {id: $id})
                SET c.title = $title,
                    c.description = $description,
                    c.damage_amount = $damage_amount,
                    c.reported_at = $reported_at,
                    c.platform = $platform
                """,
                id=case.id,
                title=case.title,
                description=case.description,
                damage_amount=case.damage_amount,
                reported_at=case.reported_at.isoformat(),
                platform=case.platform
            )

            # 문구와 사례 연결
            for phrase_id in case.phrases_used:
                await session.run(
                    """
                    MATCH (c:ScamCase {id: $case_id})
                    MATCH (p:ScamPhrase {id: $phrase_id})
                    MERGE (c)-[:USED_PHRASE]->(p)
                    """,
                    case_id=case.id,
                    phrase_id=phrase_id
                )

    # ==================== RAG 조회 ====================

    async def search_by_keywords(self, keywords: list[str]) -> RAGResult:
        """키워드로 관련 스캠 패턴 조회"""
        if not self._driver:
            return RAGResult(
                matched_phrases=[],
                similar_cases=[],
                total_reports=0,
                risk_indicators=[]
            )

        matched_phrases = []
        similar_cases = []
        risk_indicators = []

        async with self._driver.session() as session:
            # 1. 키워드와 매칭되는 문구 조회
            for keyword in keywords:
                result = await session.run(
                    """
                    MATCH (p:ScamPhrase)
                    WHERE p.text CONTAINS $keyword
                       OR ANY(ex IN p.examples WHERE ex CONTAINS $keyword)
                    RETURN p
                    ORDER BY p.severity DESC, p.usage_count DESC
                    LIMIT 5
                    """,
                    keyword=keyword
                )
                records = await result.data()

                for record in records:
                    p = record["p"]
                    phrase = ScamPhrase(
                        id=p["id"],
                        text=p["text"],
                        category=p["category"],
                        severity=p["severity"],
                        usage_count=p["usage_count"],
                        examples=p.get("examples", [])
                    )
                    if phrase.id not in [mp.id for mp in matched_phrases]:
                        matched_phrases.append(phrase)

            # 2. 매칭된 문구와 연결된 스캠 사례 조회
            if matched_phrases:
                phrase_ids = [p.id for p in matched_phrases]
                result = await session.run(
                    """
                    MATCH (c:ScamCase)-[:USED_PHRASE]->(p:ScamPhrase)
                    WHERE p.id IN $phrase_ids
                    RETURN DISTINCT c, collect(p.id) as phrase_ids
                    ORDER BY c.damage_amount DESC
                    LIMIT 5
                    """,
                    phrase_ids=phrase_ids
                )
                records = await result.data()

                for record in records:
                    c = record["c"]
                    case = ScamCase(
                        id=c["id"],
                        title=c["title"],
                        description=c["description"],
                        phrases_used=record["phrase_ids"],
                        damage_amount=c.get("damage_amount"),
                        reported_at=datetime.fromisoformat(c["reported_at"]) if c.get("reported_at") else datetime.now(),
                        platform=c.get("platform", "unknown")
                    )
                    similar_cases.append(case)

            # 3. 전체 신고 건수
            result = await session.run("MATCH (c:ScamCase) RETURN count(c) as count")
            record = await result.single()
            total_reports = record["count"] if record else 0

            # 4. 위험 지표 생성
            for phrase in matched_phrases:
                if phrase.severity >= 8:
                    risk_indicators.append(f"고위험 패턴 '{phrase.text}' 감지 (위험도: {phrase.severity}/10)")
                if phrase.usage_count >= 50:
                    risk_indicators.append(f"'{phrase.text}' - {phrase.usage_count}건의 스캠에서 사용됨")

        return RAGResult(
            matched_phrases=matched_phrases,
            similar_cases=similar_cases,
            total_reports=total_reports,
            risk_indicators=risk_indicators
        )

    async def search_by_text(self, text: str) -> RAGResult:
        """전체 텍스트로 검색 (키워드 자동 추출)"""
        # 간단한 키워드 추출
        keywords = []
        keyword_list = [
            "사랑", "보고싶", "투자", "돈", "송금", "급", "빨리", "비밀",
            "결혼", "만나", "아파", "사고", "병원", "도와", "계좌", "코인",
            "비트코인", "수익", "기회", "영상통화", "얼굴", "사진"
        ]

        for kw in keyword_list:
            if kw in text:
                keywords.append(kw)

        if not keywords:
            # 키워드가 없으면 텍스트를 공백으로 분리
            words = text.split()
            keywords = [w for w in words if len(w) >= 2][:5]

        return await self.search_by_keywords(keywords)

    # ==================== 통계 ====================

    async def get_stats(self) -> dict:
        """통계 조회"""
        if not self._driver:
            return {"phrases": 0, "cases": 0}

        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (p:ScamPhrase) WITH count(p) as phrases
                MATCH (c:ScamCase) WITH phrases, count(c) as cases
                RETURN phrases, cases
                """
            )
            record = await result.single()
            return {
                "phrases": record["phrases"] if record else 0,
                "cases": record["cases"] if record else 0
            }
