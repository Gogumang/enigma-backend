from abc import ABC, abstractmethod

from src.domain.fraud import FraudSource


class FraudExternalPort(ABC):
    """외부 사기 조회 API 포트 (Driven Port)"""

    @abstractmethod
    async def search(self, fraud_type: str, value: str) -> FraudSource:
        """사기 이력 조회. FraudSource 반환."""
        ...
