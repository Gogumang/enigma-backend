"""
Neo4j 기반 사기 신고 리포지토리
- 사기 신고 저장 및 조회
- 사기꾼 식별 정보(전화번호, 계좌, SNS, URL) 관리
"""
import logging
import uuid
from typing import Optional

from neo4j import AsyncGraphDatabase

from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class ScamReportRepository:
    """Neo4j 기반 사기 신고 리포지토리"""

    def __init__(self):
        settings = get_settings()
        self.uri = settings.neo4j_uri
        self.username = settings.neo4j_username
        self.password = settings.neo4j_password
        self._driver = None

    async def initialize(self) -> None:
        """Neo4j 연결 초기화"""
        if not self.uri:
            logger.warning("Neo4j URI not configured for scam report repository")
            return

        try:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            async with self._driver.session() as session:
                await session.run("RETURN 1")
            logger.info("ScamReportRepository connected")

            await self._create_indexes()

        except Exception as e:
            logger.error(f"Failed to initialize ScamReportRepository: {e}")
            self._driver = None

    async def _create_indexes(self) -> None:
        """인덱스 생성"""
        if not self._driver:
            return

        async with self._driver.session() as session:
            await session.run(
                "CREATE INDEX scam_report_id IF NOT EXISTS FOR (r:ScamReport) ON (r.id)"
            )
            await session.run(
                "CREATE INDEX scammer_info_value IF NOT EXISTS FOR (s:ScammerInfo) ON (s.value)"
            )

    async def close(self) -> None:
        """연결 종료"""
        if self._driver:
            await self._driver.close()

    def is_connected(self) -> bool:
        return self._driver is not None

    async def save_report(self, report_data: dict) -> str:
        """사기 신고 저장"""
        if not self._driver:
            raise RuntimeError("Neo4j not connected")

        report_id = str(uuid.uuid4())

        async with self._driver.session() as session:
            # 신고 노드 생성
            await session.run(
                """
                MERGE (r:ScamReport {id: $report_id})
                SET r.overallScore = $overall_score,
                    r.deepfakeScore = $deepfake_score,
                    r.chatScore = $chat_score,
                    r.fraudScore = $fraud_score,
                    r.urlScore = $url_score,
                    r.profileScore = $profile_score,
                    r.reasons = $reasons,
                    r.details = $details,
                    r.reportedAt = datetime()
                """,
                report_id=report_id,
                overall_score=report_data.get("overallScore", 0),
                deepfake_score=report_data.get("deepfakeScore", 0),
                chat_score=report_data.get("chatScore", 0),
                fraud_score=report_data.get("fraudScore", 0),
                url_score=report_data.get("urlScore", 0),
                profile_score=report_data.get("profileScore", 0),
                reasons=report_data.get("reasons", []),
                details=report_data.get("details", ""),
            )

            # 식별 정보 노드 연결
            identifiers = report_data.get("identifiers", [])
            for identifier in identifiers:
                id_type = identifier.get("type", "")
                id_value = identifier.get("value", "")
                if id_type and id_value:
                    await session.run(
                        """
                        MATCH (r:ScamReport {id: $report_id})
                        MERGE (s:ScammerInfo {type: $id_type, value: $id_value})
                        MERGE (r)-[:IDENTIFIED]->(s)
                        SET s.updatedAt = datetime(),
                            s.reportCount = coalesce(s.reportCount, 0) + 1
                        """,
                        report_id=report_id,
                        id_type=id_type,
                        id_value=id_value,
                    )

        return report_id

    async def get_report(self, report_id: str) -> Optional[dict]:
        """신고 조회"""
        if not self._driver:
            return None

        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (r:ScamReport {id: $report_id})
                OPTIONAL MATCH (r)-[:IDENTIFIED]->(s:ScammerInfo)
                RETURN r, collect(s) as identifiers
                """,
                report_id=report_id,
            )
            record = await result.single()

            if not record:
                return None

            report_node = record["r"]
            identifiers_nodes = record["identifiers"]

            return {
                "id": report_node["id"],
                "overallScore": report_node.get("overallScore", 0),
                "deepfakeScore": report_node.get("deepfakeScore", 0),
                "chatScore": report_node.get("chatScore", 0),
                "fraudScore": report_node.get("fraudScore", 0),
                "urlScore": report_node.get("urlScore", 0),
                "profileScore": report_node.get("profileScore", 0),
                "reasons": list(report_node.get("reasons", [])),
                "details": report_node.get("details", ""),
                "reportedAt": str(report_node.get("reportedAt", "")),
                "identifiers": [
                    {"type": s["type"], "value": s["value"]}
                    for s in identifiers_nodes
                    if s is not None
                ],
            }

    async def find_by_identifier(self, id_type: str, id_value: str) -> list[dict]:
        """식별자로 기존 신고 검색"""
        if not self._driver:
            return []

        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (s:ScammerInfo {type: $id_type, value: $id_value})<-[:IDENTIFIED]-(r:ScamReport)
                RETURN r
                ORDER BY r.reportedAt DESC
                LIMIT 20
                """,
                id_type=id_type,
                id_value=id_value,
            )
            records = await result.data()

            return [
                {
                    "id": record["r"]["id"],
                    "overallScore": record["r"].get("overallScore", 0),
                    "reportedAt": str(record["r"].get("reportedAt", "")),
                }
                for record in records
            ]
