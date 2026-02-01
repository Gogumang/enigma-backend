import io
import logging
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from src.domain.deepfake import DeepfakeAnalysis, MediaType
from src.infrastructure.ai import ImageQualityService
from src.infrastructure.ai.deepfake_explainer import get_deepfake_explainer_service
from src.infrastructure.external import SightengineService, OpenAIService
from src.shared.exceptions import ValidationException

logger = logging.getLogger(__name__)


def convert_to_jpeg(image_data: bytes) -> bytes:
    """이미지를 JPEG로 변환 (GIF, PNG, WebP 등 지원)"""
    try:
        img = Image.open(io.BytesIO(image_data))

        # GIF인 경우 첫 프레임만 사용
        if img.format == 'GIF':
            img.seek(0)

        # RGBA인 경우 RGB로 변환 (JPEG은 알파 채널 미지원)
        if img.mode in ('RGBA', 'LA', 'P'):
            # 흰색 배경으로 변환
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # JPEG로 저장
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=95)
        return output.getvalue()
    except Exception as e:
        logger.warning(f"이미지 변환 실패, 원본 사용: {e}")
        return image_data

# 이미지 품질 서비스 (싱글톤)
_image_quality_service = ImageQualityService()


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
    # 이미지 품질 관련 필드
    image_quality: str = "high"  # high, medium, low
    blur_score: float = 100.0  # 0-100 (높을수록 선명)
    quality_warning: str | None = None
    is_reliable: bool = True  # 분석 신뢰도가 충분한지


class AnalyzeImageUseCase:
    """이미지 딥페이크 분석 유스케이스"""

    def __init__(self, sightengine: SightengineService, openai: Optional[OpenAIService] = None):
        self.sightengine = sightengine
        self.openai = openai

    async def execute(self, image_data: bytes) -> DeepfakeAnalysisResult:
        """이미지 분석 실행"""
        if not image_data:
            raise ValidationException("이미지 데이터가 필요합니다")

        # 0. GIF/PNG/WebP 등을 JPEG로 변환
        image_data = convert_to_jpeg(image_data)

        # 1. 이미지 품질 체크 (blur detection)
        quality_result = _image_quality_service.analyze(image_data)

        # 2. EfficientViT 기반 딥페이크 탐지 시도 (히트맵 + 정확한 좌표)
        explainer_result = None
        try:
            explainer = get_deepfake_explainer_service()
            if explainer.is_available():
                logger.info("Using EfficientViT DeepfakeExplainer for analysis")
                explainer_result = explainer.analyze(image_data)
        except Exception as e:
            logger.warning(f"DeepfakeExplainer failed, falling back: {e}")

        # 3. EfficientViT 결과가 있으면 사용
        if explainer_result:
            # EfficientViT에서 얻은 결과로 기본 정보 설정
            is_deepfake = explainer_result.is_deepfake
            confidence = explainer_result.confidence

            # 마커 변환 (DetectionMarker -> dict)
            markers = [
                {
                    "id": m.id,
                    "x": m.x,
                    "y": m.y,
                    "label": m.label,
                    "description": m.description,
                    "intensity": m.intensity,
                }
                for m in explainer_result.markers
            ]

            # OpenAI로 마커별 상세 설명 생성
            if self.openai:
                try:
                    ai_analysis = await self.openai.analyze_deepfake_with_markers(
                        image_data, markers, is_deepfake, confidence
                    )
                    # AI 설명으로 마커 description 업데이트
                    for marker in markers:
                        marker_id = str(marker["id"])
                        if marker_id in ai_analysis.get("descriptions", {}):
                            marker["description"] = ai_analysis["descriptions"][marker_id]
                    overall_assessment = ai_analysis.get("overall_assessment", "")
                except Exception as e:
                    logger.warning(f"OpenAI analysis failed: {e}")
                    overall_assessment = f"딥페이크 확률 {confidence:.1f}%로 분석되었습니다."
            else:
                overall_assessment = f"딥페이크 확률 {confidence:.1f}%로 분석되었습니다."

            # 위험 수준 결정
            if confidence >= 70:
                risk_level = "high"
            elif confidence >= 40:
                risk_level = "medium"
            else:
                risk_level = "low"

            result = DeepfakeAnalysisResult(
                is_deepfake=is_deepfake,
                confidence=confidence,
                risk_level=risk_level,
                media_type="image",
                message=f"딥페이크 {'의심' if is_deepfake else '가능성 낮음'} ({confidence:.1f}%)",
                details={"heatmap_base64": explainer_result.heatmap_base64},
                markers=markers,
                overall_assessment=overall_assessment,
            )

        # 4. Sightengine API 사용 (EfficientViT 사용 불가 시)
        elif self.sightengine.is_configured():
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

        # 5. 시뮬레이션 결과 (API 키 없을 때)
        else:
            result = self._simulate_result(MediaType.IMAGE)
            if self.openai:
                ai_analysis = await self.openai.analyze_deepfake_image(
                    image_data, result.is_deepfake, result.confidence
                )
                result.analysis_reasons = ai_analysis.get("analysis_reasons", [])
                result.markers = ai_analysis.get("markers", [])
                result.technical_indicators = ai_analysis.get("technical_indicators", [])
                result.overall_assessment = ai_analysis.get("overall_assessment", "")

        # 품질 정보 추가
        result.image_quality = quality_result.quality_level.value
        result.blur_score = quality_result.blur_score
        result.quality_warning = quality_result.warning_message
        result.is_reliable = quality_result.is_reliable

        # 품질이 낮으면 메시지에 경고 추가
        if not quality_result.is_reliable:
            result.message = f"[⚠️ 저화질 이미지] {result.message}"

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
