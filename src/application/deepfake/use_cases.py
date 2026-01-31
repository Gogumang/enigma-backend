import logging
from dataclasses import dataclass, field
from typing import Optional

from src.domain.deepfake import DeepfakeAnalysis, MediaType
from src.infrastructure.external import SightengineService, OpenAIService
from src.shared.exceptions import ValidationException

logger = logging.getLogger(__name__)


@dataclass
class DeepfakeAnalysisResult:
    """딥페이크 분석 결과 DTO"""
    is_deepfake: bool
    confidence: float
    risk_level: str
    media_type: str
    message: str
    details: dict
    analysis_reasons: list = field(default_factory=list)
    markers: list = field(default_factory=list)
    technical_indicators: list = field(default_factory=list)
    overall_assessment: str = ""


class AnalyzeImageUseCase:
    """이미지 딥페이크 분석 유스케이스"""

    def __init__(self, sightengine: SightengineService, openai: Optional[OpenAIService] = None):
        self.sightengine = sightengine
        self.openai = openai

    async def execute(self, image_data: bytes) -> DeepfakeAnalysisResult:
        """이미지 분석 실행"""
        if not image_data:
            raise ValidationException("이미지 데이터가 필요합니다")

        if not self.sightengine.is_configured():
            # API 키가 없으면 시뮬레이션 결과 반환
            result = self._simulate_result(MediaType.IMAGE)
            # OpenAI로 상세 분석
            if self.openai:
                ai_analysis = await self.openai.analyze_deepfake_image(
                    image_data, result.is_deepfake, result.confidence
                )
                result.analysis_reasons = ai_analysis.get("analysis_reasons", [])
                result.markers = ai_analysis.get("markers", [])
                result.technical_indicators = ai_analysis.get("technical_indicators", [])
                result.overall_assessment = ai_analysis.get("overall_assessment", "")
            return result

        analysis = await self.sightengine.analyze_image(image_data)
        result = self._to_result(analysis)

        # OpenAI로 상세 분석
        if self.openai:
            ai_analysis = await self.openai.analyze_deepfake_image(
                image_data, result.is_deepfake, result.confidence
            )
            result.analysis_reasons = ai_analysis.get("analysis_reasons", [])
            result.markers = ai_analysis.get("markers", [])
            result.technical_indicators = ai_analysis.get("technical_indicators", [])
            result.overall_assessment = ai_analysis.get("overall_assessment", "")

        return result

    async def execute_from_url(self, url: str) -> DeepfakeAnalysisResult:
        """URL 이미지 분석 실행"""
        if not url:
            raise ValidationException("이미지 URL이 필요합니다")

        if not self.sightengine.is_configured():
            return self._simulate_result(MediaType.IMAGE)

        analysis = await self.sightengine.analyze_image_url(url)
        return self._to_result(analysis)

    def _to_result(self, analysis: DeepfakeAnalysis) -> DeepfakeAnalysisResult:
        if analysis.is_deepfake:
            message = f"딥페이크 의심! ({analysis.confidence:.1f}% 확률)"
        else:
            message = f"딥페이크 가능성 낮음 ({100 - analysis.confidence:.1f}% 신뢰도)"

        return DeepfakeAnalysisResult(
            is_deepfake=analysis.is_deepfake,
            confidence=analysis.confidence,
            risk_level=analysis.risk_level.value,
            media_type=analysis.media_type.value,
            message=message,
            details=analysis.details
        )

    def _simulate_result(self, media_type: MediaType) -> DeepfakeAnalysisResult:
        """API 키 없을 때 시뮬레이션 결과"""
        import random
        confidence = random.uniform(10, 40)

        return DeepfakeAnalysisResult(
            is_deepfake=False,
            confidence=confidence,
            risk_level="low",
            media_type=media_type.value,
            message="시뮬레이션 결과입니다. 실제 분석을 위해 API 키를 설정해주세요.",
            details={"simulation": True}
        )


class AnalyzeVideoUseCase:
    """비디오 딥페이크 분석 유스케이스"""

    def __init__(self, sightengine: SightengineService):
        self.sightengine = sightengine

    async def execute(self, video_data: bytes) -> DeepfakeAnalysisResult:
        """비디오 분석 실행"""
        if not video_data:
            raise ValidationException("비디오 데이터가 필요합니다")

        if not self.sightengine.is_configured():
            return self._simulate_result()

        analysis = await self.sightengine.analyze_video(video_data)
        return self._to_result(analysis)

    def _to_result(self, analysis: DeepfakeAnalysis) -> DeepfakeAnalysisResult:
        if analysis.is_deepfake:
            message = f"딥페이크 비디오 의심! ({analysis.confidence:.1f}% 확률)"
        else:
            message = f"딥페이크 가능성 낮음 ({100 - analysis.confidence:.1f}% 신뢰도)"

        return DeepfakeAnalysisResult(
            is_deepfake=analysis.is_deepfake,
            confidence=analysis.confidence,
            risk_level=analysis.risk_level.value,
            media_type=analysis.media_type.value,
            message=message,
            details=analysis.details
        )

    def _simulate_result(self) -> DeepfakeAnalysisResult:
        import random
        confidence = random.uniform(10, 40)

        return DeepfakeAnalysisResult(
            is_deepfake=False,
            confidence=confidence,
            risk_level="low",
            media_type="video",
            message="시뮬레이션 결과입니다. 실제 분석을 위해 API 키를 설정해주세요.",
            details={"simulation": True}
        )
