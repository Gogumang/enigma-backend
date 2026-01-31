import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from src.domain.scammer import ScammerEntity, ScammerRepository

logger = logging.getLogger(__name__)


class JsonScammerRepository(ScammerRepository):
    """JSON 파일 기반 스캐머 리포지토리 구현"""

    def __init__(self, file_path: str = "data/scammer-faces.json"):
        self.file_path = Path(file_path)
        self._cache: list[ScammerEntity] = []
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """리포지토리 초기화"""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if self.file_path.exists():
            await self._load_from_file()
        else:
            await self._save_to_file()

        logger.info(f"Scammer repository initialized with {len(self._cache)} records")

    async def _load_from_file(self) -> None:
        """파일에서 데이터 로드"""
        try:
            async with self._lock:
                with open(self.file_path, encoding="utf-8") as f:
                    data = json.load(f)
                    self._cache = [ScammerEntity.from_dict(d) for d in data]
        except Exception as e:
            logger.error(f"Failed to load scammer data: {e}")
            self._cache = []

    async def _save_to_file(self) -> None:
        """파일에 데이터 저장"""
        try:
            async with self._lock:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    data = [s.to_dict() for s in self._cache]
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save scammer data: {e}")

    async def save(self, scammer: ScammerEntity) -> ScammerEntity:
        """스캐머 저장"""
        # 기존 스캐머 업데이트 또는 새로 추가
        existing_idx = next(
            (i for i, s in enumerate(self._cache) if s.id == scammer.id),
            None
        )

        if existing_idx is not None:
            self._cache[existing_idx] = scammer
        else:
            self._cache.append(scammer)

        await self._save_to_file()
        return scammer

    async def find_by_id(self, scammer_id: str) -> ScammerEntity | None:
        """ID로 스캐머 조회"""
        return next((s for s in self._cache if s.id == scammer_id), None)

    async def find_all(self) -> list[ScammerEntity]:
        """모든 스캐머 조회"""
        return self._cache.copy()

    async def find_by_face_embedding(
        self,
        embedding: list[float],
        threshold: float = 0.6
    ) -> list[tuple[ScammerEntity, float]]:
        """얼굴 임베딩으로 유사한 스캐머 찾기"""
        results = []
        query_embedding = np.array(embedding)

        for scammer in self._cache:
            scammer_embedding = np.array(scammer.face_embedding)
            distance = float(np.linalg.norm(query_embedding - scammer_embedding))

            if distance < threshold:
                results.append((scammer, distance))

        # 거리 오름차순 정렬 (가장 유사한 것 먼저)
        results.sort(key=lambda x: x[1])
        return results

    async def count(self) -> int:
        """스캐머 수 조회"""
        return len(self._cache)

    async def delete(self, scammer_id: str) -> bool:
        """스캐머 삭제"""
        original_len = len(self._cache)
        self._cache = [s for s in self._cache if s.id != scammer_id]

        if len(self._cache) < original_len:
            await self._save_to_file()
            return True
        return False
