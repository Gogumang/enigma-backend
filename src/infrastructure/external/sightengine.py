import logging

import httpx

from src.domain.deepfake import DeepfakeAnalysis, MediaType
from src.shared.config import get_settings
from src.shared.exceptions import ExternalServiceException

logger = logging.getLogger(__name__)


class SightengineService:
    """Sightengine 딥페이크 탐지 서비스"""

    BASE_URL = "https://api.sightengine.com/1.0"

    def __init__(self):
        settings = get_settings()
        self.api_user = settings.sightengine_api_user
        self.api_secret = settings.sightengine_api_secret

    def is_configured(self) -> bool:
        return bool(self.api_user and self.api_secret)

    async def analyze_image(self, image_data: bytes) -> DeepfakeAnalysis:
        """이미지 딥페이크 분석"""
        if not self.is_configured():
            raise ExternalServiceException("API credentials not configured", "Sightengine")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/check.json",
                    data={
                        "models": "deepfake,genai",
                        "api_user": self.api_user,
                        "api_secret": self.api_secret,
                    },
                    files={"media": ("image.jpg", image_data, "image/jpeg")}
                )

                if response.status_code != 200:
                    raise ExternalServiceException(
                        f"API returned {response.status_code}",
                        "Sightengine"
                    )

                result = response.json()
                return self._parse_image_result(result)

        except httpx.RequestError as e:
            logger.error(f"Sightengine request failed: {e}")
            raise ExternalServiceException(str(e), "Sightengine") from e

    async def analyze_image_url(self, url: str) -> DeepfakeAnalysis:
        """URL 이미지 딥페이크 분석"""
        if not self.is_configured():
            raise ExternalServiceException("API credentials not configured", "Sightengine")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}/check.json",
                    params={
                        "models": "deepfake,genai",
                        "api_user": self.api_user,
                        "api_secret": self.api_secret,
                        "url": url
                    }
                )

                if response.status_code != 200:
                    raise ExternalServiceException(
                        f"API returned {response.status_code}",
                        "Sightengine"
                    )

                result = response.json()
                return self._parse_image_result(result)

        except httpx.RequestError as e:
            logger.error(f"Sightengine request failed: {e}")
            raise ExternalServiceException(str(e), "Sightengine") from e

    async def analyze_video(self, video_data: bytes) -> DeepfakeAnalysis:
        """비디오 딥페이크 분석"""
        if not self.is_configured():
            raise ExternalServiceException("API credentials not configured", "Sightengine")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/video/check.json",
                    data={
                        "models": "deepfake",
                        "api_user": self.api_user,
                        "api_secret": self.api_secret,
                    },
                    files={"media": ("video.mp4", video_data, "video/mp4")}
                )

                if response.status_code != 200:
                    raise ExternalServiceException(
                        f"API returned {response.status_code}",
                        "Sightengine"
                    )

                result = response.json()
                return self._parse_video_result(result)

        except httpx.RequestError as e:
            logger.error(f"Sightengine video request failed: {e}")
            raise ExternalServiceException(str(e), "Sightengine") from e

    def _parse_image_result(self, result: dict) -> DeepfakeAnalysis:
        """이미지 분석 결과 파싱 (deepfake + genai)"""
        # deepfake 점수 (얼굴 조작)
        deepfake_score = float(result.get("type", {}).get("deepfake", 0)) * 100

        # genai 점수 (AI 생성 이미지 - DALL-E, Midjourney, Stable Diffusion 등)
        genai_score = float(result.get("type", {}).get("ai_generated", 0)) * 100

        # 둘 중 높은 점수 사용
        confidence = max(deepfake_score, genai_score)

        # 상세 정보에 각 점수 포함
        details = {
            **result,
            "deepfake_score": deepfake_score,
            "genai_score": genai_score,
            "detection_type": "genai" if genai_score > deepfake_score else "deepfake",
        }

        return DeepfakeAnalysis.create(
            is_deepfake=confidence >= 50,
            confidence=confidence,
            media_type=MediaType.IMAGE,
            details=details
        )

    def _parse_video_result(self, result: dict) -> DeepfakeAnalysis:
        """비디오 분석 결과 파싱"""
        frames = result.get("data", {}).get("frames", [])

        if not frames:
            return DeepfakeAnalysis.create(
                is_deepfake=False,
                confidence=0,
                media_type=MediaType.VIDEO,
                details=result
            )

        # 프레임별 deepfake 점수 평균
        scores = [
            f.get("type", {}).get("deepfake", 0) * 100
            for f in frames
        ]
        avg_confidence = sum(scores) / len(scores) if scores else 0

        return DeepfakeAnalysis.create(
            is_deepfake=avg_confidence >= 50,
            confidence=avg_confidence,
            media_type=MediaType.VIDEO,
            details={"frames_analyzed": len(frames), "frame_scores": scores}
        )
