"""
무료 역이미지 검색 서비스
- 이미지를 무료 호스팅에 업로드
- 각 검색 엔진의 역이미지 검색 URL 생성
"""
import base64
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ReverseImageSearchResult:
    """역이미지 검색 결과"""
    success: bool
    image_url: str | None = None
    search_links: list[dict] | None = None
    error: str | None = None


class ReverseImageSearchService:
    """무료 역이미지 검색 서비스"""

    def __init__(self):
        self.imgbb_url = "https://api.imgbb.com/1/upload"
        # imgbb 무료 API 키 (익명 업로드용, 공개 키)
        self.imgbb_key = "7a1a88f3c698393738315e07c tried95"

    async def upload_and_get_search_links(
        self,
        image_data: bytes
    ) -> ReverseImageSearchResult:
        """이미지 업로드 후 검색 링크 생성"""
        try:
            # 1. 이미지를 0x0.st에 업로드 (무료, API 키 불필요)
            image_url = await self._upload_to_0x0(image_data)

            if not image_url:
                # 백업: catbox.moe 사용
                image_url = await self._upload_to_catbox(image_data)

            if not image_url:
                return ReverseImageSearchResult(
                    success=False,
                    error="이미지 업로드 실패"
                )

            # 2. 각 검색 엔진 URL 생성
            search_links = self._generate_search_links(image_url)

            logger.info(f"Reverse image search links generated for: {image_url}")

            return ReverseImageSearchResult(
                success=True,
                image_url=image_url,
                search_links=search_links
            )

        except Exception as e:
            logger.error(f"Reverse image search failed: {e}")
            return ReverseImageSearchResult(
                success=False,
                error=str(e)
            )

    async def _upload_to_0x0(self, image_data: bytes) -> str | None:
        """0x0.st에 이미지 업로드 (무료, 익명)"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"file": ("image.jpg", image_data, "image/jpeg")}
                response = await client.post("https://0x0.st", files=files)

                if response.status_code == 200:
                    return response.text.strip()

        except Exception as e:
            logger.warning(f"0x0.st upload failed: {e}")

        return None

    async def _upload_to_catbox(self, image_data: bytes) -> str | None:
        """catbox.moe에 이미지 업로드 (무료, 익명)"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"fileToUpload": ("image.jpg", image_data, "image/jpeg")}
                data = {"reqtype": "fileupload"}
                response = await client.post(
                    "https://catbox.moe/user/api.php",
                    files=files,
                    data=data
                )

                if response.status_code == 200 and response.text.startswith("https://"):
                    return response.text.strip()

        except Exception as e:
            logger.warning(f"catbox.moe upload failed: {e}")

        return None

    def _generate_search_links(self, image_url: str) -> list[dict]:
        """각 검색 엔진별 역이미지 검색 URL 생성"""
        from urllib.parse import quote

        encoded_url = quote(image_url, safe='')

        return [
            {
                "platform": "google",
                "name": "Google 이미지 검색",
                "url": f"https://lens.google.com/uploadbyurl?url={encoded_url}",
                "icon": "🔍"
            },
            {
                "platform": "yandex",
                "name": "Yandex (얼굴 검색 강력)",
                "url": f"https://yandex.com/images/search?rpt=imageview&url={encoded_url}",
                "icon": "🔎"
            },
            {
                "platform": "bing",
                "name": "Bing 이미지 검색",
                "url": f"https://www.bing.com/images/search?view=detailv2&iss=sbi&form=SBIVSP&sbisrc=UrlPaste&q=imgurl:{encoded_url}",
                "icon": "🔷"
            },
            {
                "platform": "tineye",
                "name": "TinEye",
                "url": f"https://tineye.com/search?url={encoded_url}",
                "icon": "👁️"
            }
        ]
