from abc import ABC, abstractmethod

from .entity import ScammerEntity


class ScammerRepository(ABC):
    """스캐머 리포지토리 인터페이스 (포트)"""

    @abstractmethod
    async def save(self, scammer: ScammerEntity) -> ScammerEntity:
        """스캐머 저장"""
        pass

    @abstractmethod
    async def find_by_id(self, scammer_id: str) -> ScammerEntity | None:
        """ID로 스캐머 조회"""
        pass

    @abstractmethod
    async def find_all(self) -> list[ScammerEntity]:
        """모든 스캐머 조회"""
        pass

    @abstractmethod
    async def find_by_face_embedding(
        self,
        embedding: list[float],
        threshold: float = 0.6
    ) -> list[tuple[ScammerEntity, float]]:
        """얼굴 임베딩으로 유사한 스캐머 찾기 (거리 포함)"""
        pass

    @abstractmethod
    async def count(self) -> int:
        """스캐머 수 조회"""
        pass

    @abstractmethod
    async def delete(self, scammer_id: str) -> bool:
        """스캐머 삭제"""
        pass
