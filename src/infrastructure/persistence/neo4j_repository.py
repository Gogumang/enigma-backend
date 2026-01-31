import logging

import numpy as np
from neo4j import AsyncGraphDatabase

from src.domain.scammer import ScammerEntity, ScammerRepository
from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class Neo4jScammerRepository(ScammerRepository):
    """Neo4j 기반 스캐머 리포지토리 구현"""

    def __init__(self):
        settings = get_settings()
        self.uri = settings.neo4j_uri
        self.username = settings.neo4j_username
        self.password = settings.neo4j_password
        self._driver = None

    async def initialize(self) -> None:
        """Neo4j 연결 초기화"""
        if not self.uri:
            logger.warning("Neo4j URI not configured")
            return

        try:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # 연결 테스트
            async with self._driver.session() as session:
                await session.run("RETURN 1")
            logger.info("Connected to Neo4j")

            # 인덱스 생성
            await self._create_indexes()

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self._driver = None

    async def _create_indexes(self) -> None:
        """인덱스 생성"""
        if not self._driver:
            return

        async with self._driver.session() as session:
            await session.run(
                "CREATE INDEX scammer_id IF NOT EXISTS FOR (s:Scammer) ON (s.id)"
            )

    async def close(self) -> None:
        """연결 종료"""
        if self._driver:
            await self._driver.close()

    def is_connected(self) -> bool:
        return self._driver is not None

    async def save(self, scammer: ScammerEntity) -> ScammerEntity:
        """스캐머 저장"""
        if not self._driver:
            raise RuntimeError("Neo4j not connected")

        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (s:Scammer {id: $id})
                SET s.name = $name,
                    s.face_embedding = $embedding,
                    s.report_count = $report_count,
                    s.source = $source,
                    s.reported_at = $reported_at,
                    s.updated_at = $updated_at
                """,
                id=scammer.id,
                name=scammer.name,
                embedding=scammer.face_embedding,
                report_count=scammer.report_count,
                source=scammer.source,
                reported_at=scammer.reported_at.isoformat(),
                updated_at=scammer.updated_at.isoformat()
            )

        return scammer

    async def find_by_id(self, scammer_id: str) -> ScammerEntity | None:
        """ID로 스캐머 조회"""
        if not self._driver:
            return None

        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (s:Scammer {id: $id}) RETURN s",
                id=scammer_id
            )
            record = await result.single()

            if record:
                return self._record_to_entity(record["s"])

        return None

    async def find_all(self) -> list[ScammerEntity]:
        """모든 스캐머 조회"""
        if not self._driver:
            return []

        async with self._driver.session() as session:
            result = await session.run("MATCH (s:Scammer) RETURN s")
            records = await result.data()

            return [self._record_to_entity(r["s"]) for r in records]

    async def find_by_face_embedding(
        self,
        embedding: list[float],
        threshold: float = 0.6
    ) -> list[tuple[ScammerEntity, float]]:
        """얼굴 임베딩으로 유사한 스캐머 찾기"""
        # Neo4j에서 모든 스캐머를 가져와 Python에서 비교
        # (Neo4j 벡터 검색은 별도 플러그인 필요)
        all_scammers = await self.find_all()
        results = []

        query_embedding = np.array(embedding)

        for scammer in all_scammers:
            scammer_embedding = np.array(scammer.face_embedding)
            distance = float(np.linalg.norm(query_embedding - scammer_embedding))

            if distance < threshold:
                results.append((scammer, distance))

        results.sort(key=lambda x: x[1])
        return results

    async def count(self) -> int:
        """스캐머 수 조회"""
        if not self._driver:
            return 0

        async with self._driver.session() as session:
            result = await session.run("MATCH (s:Scammer) RETURN count(s) as count")
            record = await result.single()
            return record["count"] if record else 0

    async def delete(self, scammer_id: str) -> bool:
        """스캐머 삭제"""
        if not self._driver:
            return False

        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (s:Scammer {id: $id}) DELETE s RETURN count(s) as deleted",
                id=scammer_id
            )
            record = await result.single()
            return record["deleted"] > 0 if record else False

    def _record_to_entity(self, record: dict) -> ScammerEntity:
        """Neo4j 레코드를 엔티티로 변환"""
        from datetime import datetime

        return ScammerEntity(
            id=record["id"],
            name=record["name"],
            face_embedding=list(record["face_embedding"]),
            report_count=record.get("report_count", 1),
            source=record.get("source"),
            reported_at=datetime.fromisoformat(record["reported_at"]) if record.get("reported_at") else datetime.now(),
            updated_at=datetime.fromisoformat(record["updated_at"]) if record.get("updated_at") else datetime.now()
        )
