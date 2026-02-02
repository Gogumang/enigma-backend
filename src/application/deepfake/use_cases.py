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
    # 알고리즘 검사 결과 (신규)
    algorithm_checks: list = field(default_factory=list)
    ensemble_details: dict = field(default_factory=dict)


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
                    "algorithm_flags": getattr(m, 'algorithm_flags', []),
                }
                for m in explainer_result.markers
            ]

            # 알고리즘 검사 결과 변환
            algorithm_checks = [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "score": c.score,
                    "description": c.description,
                }
                for c in getattr(explainer_result, 'algorithm_checks', [])
            ]

            ensemble_details = getattr(explainer_result, 'ensemble_details', {})

            # 기술적 지표 생성 (알고리즘 검사 기반)
            technical_indicators = []
            for check in algorithm_checks:
                if not check["passed"]:
                    indicator_names = {
                        "frequency_analysis": "주파수 도메인 이상",
                        "skin_texture": "피부 텍스처 불일치",
                        "color_consistency": "색상/조명 불일치",
                        "edge_artifacts": "경계 아티팩트",
                        "noise_pattern": "노이즈 패턴 이상",
                        "compression_artifacts": "압축 아티팩트 이상",
                    }
                    technical_indicators.append({
                        "name": indicator_names.get(check["name"], check["name"]),
                        "description": check["description"],
                        "score": check["score"],
                    })

            # 순수 알고리즘 기반 마커 설명 생성 (OpenAI 미사용)
            markers = self._enhance_markers_with_algorithm_info(
                markers, algorithm_checks, is_deepfake, confidence
            )

            # 알고리즘 기반 종합 평가 생성
            overall_assessment = self._generate_assessment_from_algorithms(
                is_deepfake, confidence, algorithm_checks
            )

            # 위험 수준 결정 (앙상블 점수 기반)
            if confidence >= 70:
                risk_level = "high"
            elif confidence >= 50:
                risk_level = "medium"
            else:
                risk_level = "low"

            result = DeepfakeAnalysisResult(
                is_deepfake=is_deepfake,
                confidence=confidence,
                risk_level=risk_level,
                media_type="image",
                message=f"딥페이크 {'의심' if is_deepfake else '가능성 낮음'} ({confidence:.1f}%)",
                details={
                    "heatmap_base64": explainer_result.heatmap_base64,
                    "model_confidence": ensemble_details.get("model_confidence", confidence),
                    "algorithm_score": ensemble_details.get("algorithm_score", 0),
                },
                markers=markers,
                overall_assessment=overall_assessment,
                technical_indicators=technical_indicators,
                algorithm_checks=algorithm_checks,
                ensemble_details=ensemble_details,
            )

        # 4. Sightengine API 사용 (EfficientViT 사용 불가 시)
        elif self.sightengine.is_configured():
            analysis = await self.sightengine.analyze_image(image_data)
            result = self._to_result(analysis)

            # 순수 알고리즘 기반 마커 생성 (OpenAI 미사용)
            result.markers = self._generate_algorithm_markers(
                image_data, result.is_deepfake, result.confidence
            )
            result.overall_assessment = self._generate_simple_assessment(
                result.is_deepfake, result.confidence
            )

        # 5. 시뮬레이션 결과 (API 키 없을 때)
        else:
            result = self._simulate_result(MediaType.IMAGE)
            # 순수 알고리즘 기반 마커 생성 (OpenAI 미사용)
            result.markers = self._generate_algorithm_markers(
                image_data, result.is_deepfake, result.confidence
            )
            result.overall_assessment = self._generate_simple_assessment(
                result.is_deepfake, result.confidence
            )

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

    def _generate_algorithm_markers(
        self,
        image_data: bytes,
        is_deepfake: bool,
        confidence: float
    ) -> list:
        """딥페이크 의심시에만 마커 생성"""
        # 딥페이크가 아니면 마커 없음
        if not is_deepfake:
            return []

        # 신뢰도가 낮으면 마커 없음
        if confidence < 50:
            return []

        from src.infrastructure.ai.face_landmark import get_face_landmark_service

        face_service = get_face_landmark_service()

        # 신뢰도에 따라 마커 수 조절
        if confidence >= 70:
            count = 3
        elif confidence >= 60:
            count = 2
        else:
            count = 1

        markers = face_service.get_analysis_markers(image_data, is_deepfake, count=count)

        base_intensity = confidence / 100.0

        for i, marker in enumerate(markers):
            intensity = base_intensity * (1 - i * 0.1)
            marker["intensity"] = round(intensity, 2)

            if intensity > 0.7:
                marker["description"] = f"높은 강도의 AI 조작 의심 영역 (강도: {intensity:.0%})"
            elif intensity > 0.5:
                marker["description"] = f"AI 조작 가능성 있는 영역 (강도: {intensity:.0%})"
            else:
                marker["description"] = f"AI 조작 의심 영역 (강도: {intensity:.0%})"

            marker["algorithm_flags"] = []

        return markers

    def _generate_simple_assessment(
        self,
        is_deepfake: bool,
        confidence: float
    ) -> str:
        """간단한 평가 문구 생성"""
        if is_deepfake:
            if confidence >= 70:
                return f"딥페이크 확률 {confidence:.1f}%. 이 이미지는 AI로 생성되었거나 조작되었을 가능성이 높습니다."
            elif confidence >= 50:
                return f"딥페이크 확률 {confidence:.1f}%. AI 조작 흔적이 일부 감지되었습니다. 주의가 필요합니다."
            else:
                return f"딥페이크 확률 {confidence:.1f}%. 약간의 의심 패턴이 감지되었으나 확실하지 않습니다."
        else:
            return f"분석 결과, 이 이미지는 딥페이크일 가능성이 낮습니다 ({confidence:.1f}%). 자연스러운 이미지 특성이 감지되었습니다."

    def _enhance_markers_with_algorithm_info(
        self,
        markers: list,
        algorithm_checks: list,
        is_deepfake: bool,
        confidence: float
    ) -> list:
        """알고리즘 검사 결과를 기반으로 마커 설명 강화"""
        # 영역별 관련 알고리즘 매핑
        region_algorithm_map = {
            "눈": ["color_consistency", "edge_artifacts", "noise_pattern"],
            "볼": ["skin_texture", "color_consistency", "noise_pattern"],
            "피부": ["skin_texture", "noise_pattern", "compression_artifacts"],
            "코": ["edge_artifacts", "skin_texture"],
            "입": ["edge_artifacts", "color_consistency"],
            "이마": ["skin_texture", "edge_artifacts", "frequency_analysis"],
            "턱": ["edge_artifacts", "skin_texture"],
            "헤어": ["edge_artifacts", "frequency_analysis"],
        }

        # 알고리즘 이름 -> 한글 설명
        algorithm_descriptions = {
            "frequency_analysis": "GAN 생성 주파수 패턴",
            "skin_texture": "피부 텍스처 이상",
            "color_consistency": "색상/조명 불일치",
            "edge_artifacts": "경계 블렌딩 흔적",
            "noise_pattern": "노이즈 패턴 불일치",
            "compression_artifacts": "압축 아티팩트",
        }

        # 의심 알고리즘 목록
        failed_checks = {c["name"]: c for c in algorithm_checks if not c.get("passed", True)}

        for marker in markers:
            label = marker.get("label", "")
            intensity = marker.get("intensity", 0)

            # 해당 영역과 관련된 알고리즘 찾기
            related_issues = []
            for region_key, algo_names in region_algorithm_map.items():
                if region_key in label:
                    for algo_name in algo_names:
                        if algo_name in failed_checks:
                            related_issues.append(algorithm_descriptions.get(algo_name, algo_name))
                    break

            # 설명 생성
            if is_deepfake:
                if related_issues:
                    # 관련 알고리즘 이슈가 있으면 구체적으로 설명
                    issues_text = ", ".join(related_issues[:2])
                    marker["description"] = f"AI 조작 의심 영역 - {issues_text} 감지 (강도: {intensity:.0%})"
                elif intensity > 0.7:
                    marker["description"] = f"높은 강도의 AI 생성 흔적 감지 (강도: {intensity:.0%})"
                elif intensity > 0.4:
                    marker["description"] = f"AI 조작 가능성 있는 영역 (강도: {intensity:.0%})"
                else:
                    marker["description"] = f"경미한 이상 패턴 감지 (강도: {intensity:.0%})"
            else:
                if intensity > 0.5:
                    marker["description"] = f"분석된 영역 - 약간의 이상 패턴 (강도: {intensity:.0%})"
                else:
                    marker["description"] = f"정상 범위의 영역 (강도: {intensity:.0%})"

            # algorithm_flags 업데이트
            marker["algorithm_flags"] = related_issues

        return markers

    def _generate_assessment_from_algorithms(
        self,
        is_deepfake: bool,
        confidence: float,
        algorithm_checks: list
    ) -> str:
        """알고리즘 검사 결과 기반 평가 문구 생성"""
        if not is_deepfake:
            return f"분석 결과, 이 이미지는 딥페이크일 가능성이 낮습니다 ({confidence:.1f}%). 자연스러운 이미지 특성이 감지되었습니다."

        # 의심 항목 수집
        suspicious_items = [c for c in algorithm_checks if not c.get("passed", True)]

        if not suspicious_items:
            return f"딥페이크 확률 {confidence:.1f}%로 분석되었습니다. AI 모델이 의심스러운 패턴을 감지했습니다."

        # 의심 항목별 설명
        issue_names = {
            "frequency_analysis": "GAN 생성 패턴",
            "skin_texture": "피부 텍스처 이상",
            "color_consistency": "색상/조명 불일치",
            "edge_artifacts": "얼굴 경계 아티팩트",
            "noise_pattern": "노이즈 패턴 불일치",
            "compression_artifacts": "압축 아티팩트 이상",
        }

        detected_issues = [
            issue_names.get(item["name"], item["name"])
            for item in suspicious_items[:3]
        ]

        if len(detected_issues) == 1:
            issues_text = detected_issues[0]
        elif len(detected_issues) == 2:
            issues_text = f"{detected_issues[0]} 및 {detected_issues[1]}"
        else:
            issues_text = f"{', '.join(detected_issues[:-1])} 및 {detected_issues[-1]}"

        return f"딥페이크 확률 {confidence:.1f}%. 다중 알고리즘 분석 결과, {issues_text}이(가) 감지되었습니다. 이 이미지는 AI로 생성되었거나 조작되었을 가능성이 높습니다."

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
