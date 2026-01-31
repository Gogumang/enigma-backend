"""
Neo4j 기반 스캐머 네트워크 분석 리포지토리
- 스캐머 간 관계 추적 (같은 계좌, 전화번호, 프로필 등)
- 스캠 네트워크 탐지
- 연관 스캐머 찾기
"""
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from neo4j import AsyncGraphDatabase

from src.shared.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ScammerReport:
    """스캐머 신고 정보"""
    id: str
    platform: str  # kakao, instagram, telegram, tinder, etc.
    profile_name: str
    description: str
    reported_at: datetime
    # 연결 정보 (옵션)
    phone_numbers: list[str] = field(default_factory=list)
    bank_accounts: list[str] = field(default_factory=list)
    profile_photo_hash: Optional[str] = None
    damage_amount: Optional[int] = None
    scam_patterns: list[str] = field(default_factory=list)


@dataclass
class NetworkAnalysis:
    """네트워크 분석 결과"""
    related_scammers: list[dict]
    shared_accounts: list[str]
    shared_phones: list[str]
    network_size: int
    risk_level: str
    warnings: list[str]


class ScammerNetworkRepository:
    """스캐머 네트워크 분석 리포지토리"""

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
            # 연결 테스트
            async with self._driver.session() as session:
                await session.run("RETURN 1")

            logger.info("ScammerNetworkRepository connected to Neo4j")
            await self._create_schema()
            await self._seed_sample_data()

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self._driver = None

    async def _create_schema(self) -> None:
        """스키마 및 인덱스 생성"""
        if not self._driver:
            return

        async with self._driver.session() as session:
            # 인덱스 생성
            constraints_and_indexes = [
                "CREATE INDEX scammer_id IF NOT EXISTS FOR (s:Scammer) ON (s.id)",
                "CREATE INDEX scammer_platform IF NOT EXISTS FOR (s:Scammer) ON (s.platform)",
                "CREATE INDEX account_number IF NOT EXISTS FOR (a:BankAccount) ON (a.number)",
                "CREATE INDEX phone_number IF NOT EXISTS FOR (p:Phone) ON (p.number)",
                "CREATE INDEX profile_hash IF NOT EXISTS FOR (pr:ProfilePhoto) ON (pr.hash)",
            ]

            for query in constraints_and_indexes:
                try:
                    await session.run(query)
                except Exception:
                    pass  # 이미 존재하는 경우 무시

            logger.info("Scammer network schema created")

    async def _seed_sample_data(self) -> None:
        """샘플 스캐머 네트워크 데이터 추가"""
        if not self._driver:
            return

        async with self._driver.session() as session:
            # 기존 데이터 확인
            result = await session.run("MATCH (s:Scammer) RETURN count(s) as count")
            record = await result.single()
            if record and record["count"] > 0:
                logger.info(f"Scammer network already has {record['count']} scammers")
                return

            # 샘플 데이터 추가 - 연결된 스캐머 네트워크
            sample_data = """
            // 스캐머 그룹 1: 같은 계좌 사용
            CREATE (s1:Scammer {
                id: 'scammer-001',
                platform: 'instagram',
                profile_name: 'lovely_james_92',
                description: '미군 장교 사칭, 투자 권유',
                reported_at: datetime('2024-01-15'),
                damage_amount: 5000000
            })
            CREATE (s2:Scammer {
                id: 'scammer-002',
                platform: 'kakao',
                profile_name: '제임스리',
                description: '같은 사진 사용, 송금 요청',
                reported_at: datetime('2024-01-20'),
                damage_amount: 3000000
            })
            CREATE (s3:Scammer {
                id: 'scammer-003',
                platform: 'tinder',
                profile_name: 'James_Engineer',
                description: '해외 엔지니어 사칭',
                reported_at: datetime('2024-02-01'),
                damage_amount: 8000000
            })

            // 공유 리소스
            CREATE (a1:BankAccount {number: '110-123-456789', bank: '신한은행'})
            CREATE (a2:BankAccount {number: '352-0987-6543-21', bank: '농협'})
            CREATE (p1:Phone {number: '010-1234-5678'})
            CREATE (p2:Phone {number: '+1-555-0123'})
            CREATE (photo1:ProfilePhoto {hash: 'abc123def456', description: '백인 남성 군복 사진'})

            // 관계 생성 - 같은 계좌
            CREATE (s1)-[:USED_ACCOUNT {first_seen: datetime('2024-01-15')}]->(a1)
            CREATE (s2)-[:USED_ACCOUNT {first_seen: datetime('2024-01-20')}]->(a1)
            CREATE (s3)-[:USED_ACCOUNT {first_seen: datetime('2024-02-01')}]->(a2)
            CREATE (s1)-[:USED_ACCOUNT {first_seen: datetime('2024-01-18')}]->(a2)

            // 관계 생성 - 같은 전화번호
            CREATE (s1)-[:USED_PHONE]->(p1)
            CREATE (s2)-[:USED_PHONE]->(p1)
            CREATE (s3)-[:USED_PHONE]->(p2)

            // 관계 생성 - 같은 프로필 사진
            CREATE (s1)-[:USED_PHOTO]->(photo1)
            CREATE (s2)-[:USED_PHOTO]->(photo1)
            CREATE (s3)-[:USED_PHOTO]->(photo1)

            // 스캐머 그룹 2: 다른 네트워크
            CREATE (s4:Scammer {
                id: 'scammer-004',
                platform: 'telegram',
                profile_name: '코인투자전문가',
                description: '코인 투자 사기',
                reported_at: datetime('2024-02-10'),
                damage_amount: 15000000
            })
            CREATE (s5:Scammer {
                id: 'scammer-005',
                platform: 'kakao',
                profile_name: '수익보장투자',
                description: '같은 수법의 투자 사기',
                reported_at: datetime('2024-02-15'),
                damage_amount: 20000000
            })
            CREATE (a3:BankAccount {number: '333-22-1111', bank: '카카오뱅크'})
            CREATE (p3:Phone {number: '010-9999-8888'})

            CREATE (s4)-[:USED_ACCOUNT]->(a3)
            CREATE (s5)-[:USED_ACCOUNT]->(a3)
            CREATE (s4)-[:USED_PHONE]->(p3)
            CREATE (s5)-[:USED_PHONE]->(p3)

            // 스캠 패턴 노드
            CREATE (pat1:ScamPattern {name: 'love_bombing', description: '과도한 애정 표현'})
            CREATE (pat2:ScamPattern {name: 'financial_request', description: '금전 요청'})
            CREATE (pat3:ScamPattern {name: 'investment_scam', description: '투자 사기'})

            CREATE (s1)-[:USED_PATTERN]->(pat1)
            CREATE (s1)-[:USED_PATTERN]->(pat2)
            CREATE (s2)-[:USED_PATTERN]->(pat1)
            CREATE (s2)-[:USED_PATTERN]->(pat2)
            CREATE (s3)-[:USED_PATTERN]->(pat1)
            CREATE (s4)-[:USED_PATTERN]->(pat3)
            CREATE (s5)-[:USED_PATTERN]->(pat3)
            """

            try:
                await session.run(sample_data)
                logger.info("Sample scammer network data seeded")
            except Exception as e:
                logger.warning(f"Failed to seed sample data (may already exist): {e}")

    def is_connected(self) -> bool:
        return self._driver is not None

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    # ==================== 스캐머 신고 ====================

    async def report_scammer(self, report: ScammerReport) -> str:
        """스캐머 신고 및 네트워크 연결"""
        if not self._driver:
            return ""

        async with self._driver.session() as session:
            # 1. 스캐머 노드 생성
            await session.run(
                """
                MERGE (s:Scammer {id: $id})
                SET s.platform = $platform,
                    s.profile_name = $profile_name,
                    s.description = $description,
                    s.reported_at = datetime($reported_at),
                    s.damage_amount = $damage_amount
                """,
                id=report.id,
                platform=report.platform,
                profile_name=report.profile_name,
                description=report.description,
                reported_at=report.reported_at.isoformat(),
                damage_amount=report.damage_amount
            )

            # 2. 계좌번호 연결
            for account in report.bank_accounts:
                await session.run(
                    """
                    MERGE (a:BankAccount {number: $number})
                    WITH a
                    MATCH (s:Scammer {id: $scammer_id})
                    MERGE (s)-[:USED_ACCOUNT]->(a)
                    """,
                    number=account,
                    scammer_id=report.id
                )

            # 3. 전화번호 연결
            for phone in report.phone_numbers:
                await session.run(
                    """
                    MERGE (p:Phone {number: $number})
                    WITH p
                    MATCH (s:Scammer {id: $scammer_id})
                    MERGE (s)-[:USED_PHONE]->(p)
                    """,
                    number=phone,
                    scammer_id=report.id
                )

            # 4. 프로필 사진 해시 연결
            if report.profile_photo_hash:
                await session.run(
                    """
                    MERGE (ph:ProfilePhoto {hash: $hash})
                    WITH ph
                    MATCH (s:Scammer {id: $scammer_id})
                    MERGE (s)-[:USED_PHOTO]->(ph)
                    """,
                    hash=report.profile_photo_hash,
                    scammer_id=report.id
                )

            # 5. 스캠 패턴 연결
            for pattern in report.scam_patterns:
                await session.run(
                    """
                    MERGE (pat:ScamPattern {name: $name})
                    WITH pat
                    MATCH (s:Scammer {id: $scammer_id})
                    MERGE (s)-[:USED_PATTERN]->(pat)
                    """,
                    name=pattern,
                    scammer_id=report.id
                )

            logger.info(f"Scammer reported: {report.id}")
            return report.id

    # ==================== 네트워크 분석 ====================

    async def analyze_network(self, scammer_id: str) -> NetworkAnalysis:
        """특정 스캐머의 네트워크 분석"""
        if not self._driver:
            return NetworkAnalysis(
                related_scammers=[],
                shared_accounts=[],
                shared_phones=[],
                network_size=0,
                risk_level="unknown",
                warnings=["데이터베이스 연결 안됨"]
            )

        async with self._driver.session() as session:
            # 1. 연관 스캐머 찾기 (같은 계좌/전화/사진 공유)
            result = await session.run(
                """
                MATCH (s:Scammer {id: $id})
                OPTIONAL MATCH (s)-[:USED_ACCOUNT]->(a:BankAccount)<-[:USED_ACCOUNT]-(other:Scammer)
                WHERE other.id <> s.id
                WITH s, collect(DISTINCT other) as account_related

                OPTIONAL MATCH (s)-[:USED_PHONE]->(p:Phone)<-[:USED_PHONE]-(other2:Scammer)
                WHERE other2.id <> s.id
                WITH s, account_related, collect(DISTINCT other2) as phone_related

                OPTIONAL MATCH (s)-[:USED_PHOTO]->(ph:ProfilePhoto)<-[:USED_PHOTO]-(other3:Scammer)
                WHERE other3.id <> s.id
                WITH s, account_related, phone_related, collect(DISTINCT other3) as photo_related

                WITH s, account_related + phone_related + photo_related as all_related
                UNWIND all_related as related
                WITH s, collect(DISTINCT related) as unique_related

                RETURN s, unique_related
                """,
                id=scammer_id
            )

            record = await result.single()
            if not record:
                return NetworkAnalysis(
                    related_scammers=[],
                    shared_accounts=[],
                    shared_phones=[],
                    network_size=0,
                    risk_level="unknown",
                    warnings=["스캐머를 찾을 수 없음"]
                )

            related_scammers = []
            for scammer in record["unique_related"]:
                related_scammers.append({
                    "id": scammer["id"],
                    "platform": scammer.get("platform", ""),
                    "profile_name": scammer.get("profile_name", ""),
                    "damage_amount": scammer.get("damage_amount", 0),
                })

            # 2. 공유 계좌 찾기
            result = await session.run(
                """
                MATCH (s:Scammer {id: $id})-[:USED_ACCOUNT]->(a:BankAccount)
                RETURN collect(a.number) as accounts
                """,
                id=scammer_id
            )
            record = await result.single()
            shared_accounts = record["accounts"] if record else []

            # 3. 공유 전화번호 찾기
            result = await session.run(
                """
                MATCH (s:Scammer {id: $id})-[:USED_PHONE]->(p:Phone)
                RETURN collect(p.number) as phones
                """,
                id=scammer_id
            )
            record = await result.single()
            shared_phones = record["phones"] if record else []

            # 4. 전체 네트워크 크기 계산
            network_size = len(related_scammers) + 1

            # 5. 위험도 판단
            if network_size >= 5:
                risk_level = "critical"
            elif network_size >= 3:
                risk_level = "high"
            elif network_size >= 2:
                risk_level = "medium"
            else:
                risk_level = "low"

            # 6. 경고 메시지 생성
            warnings = []
            if len(related_scammers) > 0:
                warnings.append(f"이 스캐머와 연결된 {len(related_scammers)}명의 다른 스캐머가 발견됨")
            if len(shared_accounts) > 1:
                warnings.append(f"여러 계좌({len(shared_accounts)}개)를 사용한 조직적 사기 의심")

            total_damage = sum(s.get("damage_amount", 0) or 0 for s in related_scammers)
            if total_damage > 0:
                warnings.append(f"연관 네트워크 총 피해액: {total_damage:,}원")

            return NetworkAnalysis(
                related_scammers=related_scammers,
                shared_accounts=shared_accounts,
                shared_phones=shared_phones,
                network_size=network_size,
                risk_level=risk_level,
                warnings=warnings
            )

    async def find_by_account(self, account_number: str) -> list[dict]:
        """계좌번호로 연관 스캐머 찾기"""
        if not self._driver:
            return []

        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (s:Scammer)-[:USED_ACCOUNT]->(a:BankAccount {number: $number})
                RETURN s
                ORDER BY s.reported_at DESC
                """,
                number=account_number
            )

            records = await result.data()
            return [
                {
                    "id": r["s"]["id"],
                    "platform": r["s"].get("platform", ""),
                    "profile_name": r["s"].get("profile_name", ""),
                    "description": r["s"].get("description", ""),
                    "damage_amount": r["s"].get("damage_amount", 0),
                }
                for r in records
            ]

    async def find_by_phone(self, phone_number: str) -> list[dict]:
        """전화번호로 연관 스캐머 찾기"""
        if not self._driver:
            return []

        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (s:Scammer)-[:USED_PHONE]->(p:Phone {number: $number})
                RETURN s
                ORDER BY s.reported_at DESC
                """,
                number=phone_number
            )

            records = await result.data()
            return [
                {
                    "id": r["s"]["id"],
                    "platform": r["s"].get("platform", ""),
                    "profile_name": r["s"].get("profile_name", ""),
                    "description": r["s"].get("description", ""),
                    "damage_amount": r["s"].get("damage_amount", 0),
                }
                for r in records
            ]

    async def find_by_photo_hash(self, photo_hash: str) -> list[dict]:
        """프로필 사진 해시로 연관 스캐머 찾기"""
        if not self._driver:
            return []

        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (s:Scammer)-[:USED_PHOTO]->(ph:ProfilePhoto {hash: $hash})
                RETURN s
                ORDER BY s.reported_at DESC
                """,
                hash=photo_hash
            )

            records = await result.data()
            return [
                {
                    "id": r["s"]["id"],
                    "platform": r["s"].get("platform", ""),
                    "profile_name": r["s"].get("profile_name", ""),
                    "description": r["s"].get("description", ""),
                    "damage_amount": r["s"].get("damage_amount", 0),
                }
                for r in records
            ]

    async def get_network_stats(self) -> dict:
        """전체 네트워크 통계"""
        if not self._driver:
            return {"scammers": 0, "accounts": 0, "phones": 0, "total_damage": 0}

        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (s:Scammer)
                WITH count(s) as scammer_count, sum(s.damage_amount) as total_damage

                MATCH (a:BankAccount)
                WITH scammer_count, total_damage, count(a) as account_count

                MATCH (p:Phone)
                RETURN scammer_count, total_damage, account_count, count(p) as phone_count
                """
            )

            record = await result.single()
            if not record:
                return {"scammers": 0, "accounts": 0, "phones": 0, "total_damage": 0}

            return {
                "scammers": record["scammer_count"],
                "accounts": record["account_count"],
                "phones": record["phone_count"],
                "total_damage": record["total_damage"] or 0
            }

    async def get_largest_networks(self, limit: int = 5) -> list[dict]:
        """가장 큰 스캐머 네트워크 조회"""
        if not self._driver:
            return []

        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (s:Scammer)
                OPTIONAL MATCH (s)-[:USED_ACCOUNT|USED_PHONE|USED_PHOTO]->()<-[:USED_ACCOUNT|USED_PHONE|USED_PHOTO]-(connected:Scammer)
                WITH s, count(DISTINCT connected) as connections
                ORDER BY connections DESC
                LIMIT $limit
                RETURN s.id as id, s.profile_name as name, s.platform as platform,
                       s.damage_amount as damage, connections
                """,
                limit=limit
            )

            records = await result.data()
            return [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "platform": r["platform"],
                    "damage_amount": r["damage"] or 0,
                    "connections": r["connections"]
                }
                for r in records
            ]
