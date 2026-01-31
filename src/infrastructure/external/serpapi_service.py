import base64
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ReverseImageResult:
    """역이미지 검색 결과"""
    title: str
    link: str
    source: str
    thumbnail: str | None = None


@dataclass
class SerpApiSearchResult:
    """SerpApi 검색 결과"""
    success: bool
    results: list[ReverseImageResult]
    error: str | None = None


class SerpApiService:
    """SerpApi 역이미지 검색 서비스"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"

    def is_configured(self) -> bool:
        """API 키가 설정되어 있는지 확인"""
        return bool(self.api_key)

    async def reverse_image_search(
        self,
        image_data: bytes,
        max_results: int = 10
    ) -> SerpApiSearchResult:
        """이미지로 역이미지 검색 수행 (Google Lens 사용)"""
        if not self.is_configured():
            return SerpApiSearchResult(
                success=False,
                results=[],
                error="SerpApi API 키가 설정되지 않았습니다"
            )

        try:
            # 이미지를 base64로 인코딩
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{image_base64}"

            # Google Lens API 호출
            params = {
                "engine": "google_lens",
                "url": data_url,
                "api_key": self.api_key
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()

            results = []

            # Visual matches (시각적으로 유사한 이미지)
            visual_matches = data.get("visual_matches", [])
            for match in visual_matches[:max_results]:
                results.append(ReverseImageResult(
                    title=match.get("title", ""),
                    link=match.get("link", ""),
                    source=match.get("source", ""),
                    thumbnail=match.get("thumbnail")
                ))

            # Knowledge graph (인물 정보 등)
            knowledge_graph = data.get("knowledge_graph", [])
            for kg in knowledge_graph[:3]:
                if kg.get("link"):
                    results.append(ReverseImageResult(
                        title=kg.get("title", "관련 정보"),
                        link=kg.get("link", ""),
                        source="Knowledge Graph",
                        thumbnail=kg.get("thumbnail")
                    ))

            logger.info(f"SerpApi reverse image search found {len(results)} results")

            return SerpApiSearchResult(
                success=True,
                results=results
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"SerpApi HTTP error: {e.response.status_code}")
            return SerpApiSearchResult(
                success=False,
                results=[],
                error=f"API 요청 실패: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"SerpApi error: {e}")
            return SerpApiSearchResult(
                success=False,
                results=[],
                error=str(e)
            )

    async def search_by_url(
        self,
        image_url: str,
        max_results: int = 10
    ) -> SerpApiSearchResult:
        """이미지 URL로 역이미지 검색 수행"""
        if not self.is_configured():
            return SerpApiSearchResult(
                success=False,
                results=[],
                error="SerpApi API 키가 설정되지 않았습니다"
            )

        try:
            params = {
                "engine": "google_lens",
                "url": image_url,
                "api_key": self.api_key
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()

            results = []

            visual_matches = data.get("visual_matches", [])
            for match in visual_matches[:max_results]:
                results.append(ReverseImageResult(
                    title=match.get("title", ""),
                    link=match.get("link", ""),
                    source=match.get("source", ""),
                    thumbnail=match.get("thumbnail")
                ))

            return SerpApiSearchResult(
                success=True,
                results=results
            )

        except Exception as e:
            logger.error(f"SerpApi error: {e}")
            return SerpApiSearchResult(
                success=False,
                results=[],
                error=str(e)
            )
