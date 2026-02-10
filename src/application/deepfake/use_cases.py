import asyncio
import io
import logging
from dataclasses import dataclass, field

from PIL import Image

from src.domain.deepfake import DeepfakeAnalysis, MediaType
from src.infrastructure.ai import ImageQualityService
from src.infrastructure.ai.image_quality import ImageQualityLevel
from src.infrastructure.ai.deepfake_explainer import get_deepfake_explainer_service
from src.infrastructure.external import SightengineService
from src.shared.exceptions import ValidationException

# SigLIP, GenConViT 디텍터
def get_clip_detector_safe():
    """SigLIP 디텍터 안전하게 가져오기"""
    try:
        from src.infrastructure.ai.clip_detector import get_clip_detector
        return get_clip_detector()
    except Exception as e:
        logger.warning(f"SigLIP detector not available: {e}")
        return None

def get_genconvit_detector_safe():
    """GenConViT 디텍터 안전하게 가져오기"""
    try:
        from src.infrastructure.ai.genconvit_detector import get_genconvit_detector
        return get_genconvit_detector()
    except Exception as e:
        logger.warning(f"GenConViT detector not available: {e}")
        return None

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

    def __init__(self, sightengine: SightengineService):
        self.sightengine = sightengine

    async def execute(self, image_data: bytes) -> DeepfakeAnalysisResult:
        """이미지 분석 실행"""
        if not image_data:
            raise ValidationException("이미지 데이터가 필요합니다")

        # 0. GIF/PNG/WebP 등을 JPEG로 변환
        image_data = convert_to_jpeg(image_data)

        # 0.5 SR 전처리: 저화질 이미지 Real-ESRGAN 향상
        image_data, was_enhanced = _image_quality_service.enhance(image_data)

        # 1. 병렬 분석: 품질 체크 + Sightengine + SigLIP + GenD-PE
        quality_result, sightengine_result, clip_result, explainer_result = await asyncio.gather(
            asyncio.to_thread(_image_quality_service.analyze, image_data),
            self._run_sightengine(image_data),
            self._run_clip(image_data),
            self._run_explainer(image_data),
        )

        sightengine_confidence = 0
        sightengine_genai_score = 0
        if sightengine_result:
            sightengine_confidence = sightengine_result.confidence
            sightengine_genai_score = sightengine_result.details.get("genai_score", 0)

        # 3. 앙상블: Sightengine + GenD-PE 가중 평균
        if explainer_result:
            # GenD-PE 결과
            gend_confidence = explainer_result.confidence

            # 가중 평균 앙상블 (응답한 모델만 참여)
            confidence = self._weighted_ensemble(
                gend_confidence, sightengine_confidence, sightengine_genai_score
            )
            is_deepfake = confidence >= 50

            logger.info(f"Ensemble: GenD-PE={gend_confidence:.1f}%, Sightengine={sightengine_confidence:.1f}%, GenAI={sightengine_genai_score:.1f}% -> Final={confidence:.1f}%")

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
                    "passed": bool(c.passed),
                    "score": float(c.score),
                    "description": c.description,
                }
                for c in getattr(explainer_result, 'algorithm_checks', [])
            ]

            # Sightengine genai 결과 추가
            if sightengine_genai_score > 0:
                algorithm_checks.append({
                    "name": "ai_generated",
                    "passed": sightengine_genai_score < 50,
                    "score": sightengine_genai_score,
                    "description": f"AI 생성 이미지 감지 (DALL-E, Midjourney, Stable Diffusion 등): {sightengine_genai_score:.1f}%",
                })

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
                        "ai_generated": "AI 생성 이미지",
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

            # AI 생성 이미지인 경우 메시지 다르게
            if sightengine_genai_score >= 50 and sightengine_genai_score >= gend_confidence:
                message = f"AI 생성 이미지 의심 ({confidence:.1f}%)"
            else:
                message = f"딥페이크 {'의심' if is_deepfake else '가능성 낮음'} ({confidence:.1f}%)"

            result = DeepfakeAnalysisResult(
                is_deepfake=is_deepfake,
                confidence=confidence,
                risk_level=risk_level,
                media_type="image",
                message=message,
                details={
                    "heatmap_base64": explainer_result.heatmap_base64,
                    "model_confidence": ensemble_details.get("model_confidence", gend_confidence),
                    "algorithm_score": ensemble_details.get("algorithm_score", 0),
                    "sightengine_deepfake": sightengine_confidence,
                    "sightengine_genai": sightengine_genai_score,
                    "gend_pe": gend_confidence,
                },
                markers=markers,
                overall_assessment=overall_assessment,
                technical_indicators=technical_indicators,
                algorithm_checks=algorithm_checks,
                ensemble_details=ensemble_details,
            )

        # 4. SigLIP + Sightengine 경로 (GenD-PE 사용 불가 시)
        elif sightengine_result or clip_result:
            # 사용 가능한 모델들로 앙상블 구성
            clip_confidence = clip_result.fake_score if clip_result else 0

            confidence = self._fallback_ensemble(
                sightengine_confidence, sightengine_genai_score, clip_confidence,
            )
            is_deepfake = confidence >= 50

            if confidence >= 70:
                risk_level = "high"
            elif confidence >= 50:
                risk_level = "medium"
            else:
                risk_level = "low"

            message = f"딥페이크 {'의심' if is_deepfake else '가능성 낮음'} ({confidence:.1f}%)"

            logger.info(
                f"Fallback Ensemble: Sightengine={sightengine_confidence:.1f}%, "
                f"GenAI={sightengine_genai_score:.1f}%, "
                f"SigLIP={clip_confidence:.1f}% -> Final={confidence:.1f}%"
            )

            result = DeepfakeAnalysisResult(
                is_deepfake=is_deepfake,
                confidence=confidence,
                risk_level=risk_level,
                media_type="image",
                message=message,
                details={
                    "sightengine_deepfake": sightengine_confidence,
                    "sightengine_genai": sightengine_genai_score,
                    "siglip": clip_confidence,
                    "model": "SigLIP+Sightengine Fallback Ensemble",
                },
            )

            # 6가지 알고리즘 검사 (GenD-PE 없이 독립 실행)
            algorithm_checks = await self._run_algorithm_checks_standalone(image_data)

            # Sightengine genai 결과 추가
            if sightengine_genai_score > 0:
                algorithm_checks.append({
                    "name": "ai_generated",
                    "passed": sightengine_genai_score < 50,
                    "score": sightengine_genai_score,
                    "description": f"AI 생성 이미지 감지 (DALL-E, Midjourney, Stable Diffusion 등): {sightengine_genai_score:.1f}%",
                })

            result.algorithm_checks = algorithm_checks

            # 알고리즘 검사 결과로 confidence 보정
            confidence = self._boost_confidence_by_algorithms(confidence, algorithm_checks)
            is_deepfake = confidence >= 50
            result.confidence = confidence
            result.is_deepfake = is_deepfake
            if confidence >= 70:
                result.risk_level = "high"
            elif confidence >= 50:
                result.risk_level = "medium"
            else:
                result.risk_level = "low"
            result.message = f"딥페이크 {'의심' if is_deepfake else '가능성 낮음'} ({confidence:.1f}%)"

            # 기술적 지표 생성
            for check in algorithm_checks:
                if not check["passed"]:
                    indicator_names = {
                        "frequency_analysis": "주파수 도메인 이상",
                        "skin_texture": "피부 텍스처 불일치",
                        "color_consistency": "색상/조명 불일치",
                        "edge_artifacts": "경계 아티팩트",
                        "noise_pattern": "노이즈 패턴 이상",
                        "compression_artifacts": "압축 아티팩트 이상",
                        "ai_generated": "AI 생성 이미지",
                    }
                    result.technical_indicators.append({
                        "name": indicator_names.get(check["name"], check["name"]),
                        "description": check["description"],
                        "score": check["score"],
                    })

            # 마커 및 평가 생성
            result.markers = self._generate_algorithm_markers(
                image_data, result.is_deepfake, result.confidence
            )
            result.markers = self._enhance_markers_with_algorithm_info(
                result.markers, algorithm_checks, is_deepfake, confidence
            )
            result.overall_assessment = self._generate_assessment_from_algorithms(
                result.is_deepfake, result.confidence, algorithm_checks
            )

        # 5. 시뮬레이션 결과 (모든 모델 실패)
        else:
            result = self._simulate_result(MediaType.IMAGE)

            # 6가지 알고리즘 검사 (독립 실행)
            algorithm_checks = await self._run_algorithm_checks_standalone(image_data)
            result.algorithm_checks = algorithm_checks

            # 알고리즘 검사 결과로 confidence 보정
            result.confidence = self._boost_confidence_by_algorithms(result.confidence, algorithm_checks)
            result.is_deepfake = result.confidence >= 50
            if result.confidence >= 70:
                result.risk_level = "high"
            elif result.confidence >= 50:
                result.risk_level = "medium"
            else:
                result.risk_level = "low"
            result.message = f"딥페이크 {'의심' if result.is_deepfake else '가능성 낮음'} ({result.confidence:.1f}%)"

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
                    result.technical_indicators.append({
                        "name": indicator_names.get(check["name"], check["name"]),
                        "description": check["description"],
                        "score": check["score"],
                    })

            result.markers = self._generate_algorithm_markers(
                image_data, result.is_deepfake, result.confidence
            )
            result.markers = self._enhance_markers_with_algorithm_info(
                result.markers, algorithm_checks, result.is_deepfake, result.confidence
            )
            result.overall_assessment = self._generate_assessment_from_algorithms(
                result.is_deepfake, result.confidence, algorithm_checks
            )

        # 품질 정보 추가
        result.image_quality = quality_result.quality_level.value
        result.blur_score = quality_result.blur_score
        result.quality_warning = quality_result.warning_message
        result.is_reliable = quality_result.is_reliable

        # 저화질 이미지 confidence 상한 적용
        if quality_result.quality_level == ImageQualityLevel.LOW:
            max_confidence = 60.0
            if result.confidence > max_confidence:
                result.details["original_confidence"] = result.confidence
                result.confidence = max_confidence
                # risk_level 재계산
                result.risk_level = "medium" if result.confidence >= 50 else "low"
        elif quality_result.quality_level == ImageQualityLevel.MEDIUM:
            max_confidence = 80.0
            if result.confidence > max_confidence:
                result.details["original_confidence"] = result.confidence
                result.confidence = max_confidence

        # 품질이 낮으면 메시지에 경고 추가
        if not quality_result.is_reliable:
            result.message = f"[⚠️ 저화질 이미지] {result.message}"

        return result

    async def _run_sightengine(self, image_data: bytes):
        """Sightengine API 비동기 호출"""
        if not self.sightengine.is_configured():
            return None
        try:
            logger.info("Using Sightengine API for AI/deepfake detection")
            return await self.sightengine.analyze_image(image_data)
        except Exception as e:
            logger.warning(f"Sightengine API failed: {e}")
            return None

    async def _run_clip(self, image_data: bytes):
        """SigLIP 디텍터 분석 (sync → async 변환)"""
        detector = get_clip_detector_safe()
        if not detector or not detector.is_available():
            return None
        try:
            logger.info("Using SigLIP for AI-generated image detection")
            result = await asyncio.to_thread(detector.analyze, image_data)
            logger.info(f"SigLIP result: fake_score={result.fake_score:.1f}%")
            return result
        except Exception as e:
            logger.warning(f"SigLIP detector failed: {e}")
            return None

    async def _run_explainer(self, image_data: bytes):
        """GenD-PE DeepfakeExplainer 분석 (sync → async 변환)"""
        try:
            explainer = get_deepfake_explainer_service()
            if not explainer.is_available():
                return None
            logger.info("Using GenD-PE DeepfakeExplainer for analysis")
            return await asyncio.to_thread(explainer.analyze, image_data)
        except Exception as e:
            logger.warning(f"DeepfakeExplainer failed, falling back: {e}")
            return None

    async def _run_algorithm_checks_standalone(self, image_data: bytes) -> list[dict]:
        """GenD-PE 없이 독립적으로 6가지 알고리즘 검사 실행"""
        try:
            import numpy as np
            from src.infrastructure.ai.deepfake_explainer.service import run_all_algorithm_checks

            img = Image.open(io.BytesIO(image_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image_array = np.array(img)
            checks = await asyncio.to_thread(run_all_algorithm_checks, image_array)
            return [
                {
                    "name": c.name,
                    "passed": bool(c.passed),
                    "score": float(c.score),
                    "description": c.description,
                }
                for c in checks
            ]
        except Exception as e:
            logger.warning(f"Standalone algorithm checks failed: {e}")
            return []

    @staticmethod
    def _weighted_ensemble(
        gend: float, sightengine: float, genai: float
    ) -> float:
        """가중 평균 앙상블 — 응답한 모델만 참여, 0은 미응답으로 간주
        Primary: GenD-PE(55%) + Sightengine(25%) + GenAI(20%)
        """
        candidates = [
            (gend, 0.55),
            (sightengine, 0.25),
            (genai, 0.20),
        ]
        total_weight = 0.0
        weighted_sum = 0.0
        for score, weight in candidates:
            if score > 0:
                weighted_sum += score * weight
                total_weight += weight
        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight

    @staticmethod
    def _fallback_ensemble(
        sightengine: float, genai: float, siglip: float
    ) -> float:
        """GenD-PE 없이 사용 가능한 모델만으로 앙상블
        Fallback: Sightengine(45%) + GenAI(40%) + SigLIP(15%)
        SigLIP은 보조 지표로만 활용 (Sightengine/GenAI 대비 정확도 낮음)
        """
        candidates = [
            (sightengine, 0.45),
            (genai, 0.40),
            (siglip, 0.15),
        ]
        total_weight = 0.0
        weighted_sum = 0.0
        for score, weight in candidates:
            if score > 0:
                weighted_sum += score * weight
                total_weight += weight
        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight

    @staticmethod
    def _boost_confidence_by_algorithms(
        confidence: float, algorithm_checks: list[dict]
    ) -> float:
        """알고리즘 검사 결과로 confidence 보정 (폴백 경로용)
        GenD-PE가 없을 때, 6가지 독립 알고리즘 검사 결과를 점수에 반영
        """
        if not algorithm_checks:
            return confidence

        # ai_generated 제외한 순수 이미지 분석 알고리즘만 카운트
        image_checks = [
            c for c in algorithm_checks if c["name"] != "ai_generated"
        ]
        suspicious_count = sum(
            1 for c in image_checks if not c.get("passed", True)
        )

        # 의심 알고리즘 가중 평균 점수
        suspicious_scores = [
            c["score"] for c in image_checks
            if not c.get("passed", True) and isinstance(c.get("score"), (int, float))
        ]
        avg_suspicious_score = (
            sum(suspicious_scores) / len(suspicious_scores)
            if suspicious_scores else 0
        )

        # 의심 항목 수에 따라 보정
        if suspicious_count >= 4:
            boost = 1.15
        elif suspicious_count >= 3:
            boost = 1.10
        elif suspicious_count >= 2:
            boost = 1.05
        else:
            boost = 1.0

        # 의심 알고리즘의 평균 점수가 높으면 추가 보정
        if avg_suspicious_score > 0.6 and suspicious_count >= 2:
            boost += 0.03

        boosted = min(100.0, confidence * boost)

        if boost > 1.0:
            logger.info(
                f"Algorithm boost: {confidence:.1f}% -> {boosted:.1f}% "
                f"(suspicious={suspicious_count}, avg_score={avg_suspicious_score:.2f}, boost={boost:.2f})"
            )

        return boosted

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
    """비디오 딥페이크 분석 유스케이스 (GenConViT + Sightengine 앙상블)"""

    def __init__(self, sightengine: SightengineService):
        self.sightengine = sightengine

    async def execute(self, video_data: bytes) -> DeepfakeAnalysisResult:
        """비디오 분석 실행 (GenConViT + Sightengine 앙상블)"""
        if not video_data:
            raise ValidationException("비디오 데이터가 필요합니다")

        # 병렬 분석: GenConViT + Sightengine
        genconvit_result, sightengine_result = await asyncio.gather(
            self._run_genconvit(video_data),
            self._run_sightengine_video(video_data),
        )

        genconvit_confidence = genconvit_result.confidence if genconvit_result else 0
        sightengine_confidence = sightengine_result.confidence if sightengine_result else 0

        # 가중 앙상블: GenConViT 70%, Sightengine 30% (응답한 엔진만 참여)
        candidates = [
            (genconvit_confidence, 0.70),
            (sightengine_confidence, 0.30),
        ]
        weighted_sum = sum(s * w for s, w in candidates if s > 0)
        total_weight = sum(w for s, w in candidates if s > 0)
        confidence = weighted_sum / total_weight if total_weight > 0 else 0

        # 둘 다 실패하면 시뮬레이션 결과
        if confidence == 0 and not genconvit_result and not sightengine_result:
            return self._simulate_result()

        is_deepfake = confidence >= 50

        # 위험 수준 결정
        if confidence >= 70:
            risk_level = "high"
        elif confidence >= 50:
            risk_level = "medium"
        else:
            risk_level = "low"

        if is_deepfake:
            message = f"딥페이크 비디오 의심! ({confidence:.1f}% 확률)"
        else:
            message = f"딥페이크 가능성 낮음 ({100 - confidence:.1f}% 신뢰도)"

        logger.info(f"Video Ensemble: GenConViT={genconvit_confidence:.1f}%, Sightengine={sightengine_confidence:.1f}% -> Final={confidence:.1f}%")

        return DeepfakeAnalysisResult(
            is_deepfake=is_deepfake,
            confidence=confidence,
            risk_level=risk_level,
            media_type="video",
            message=message,
            details={
                "genconvit": genconvit_confidence,
                "sightengine": sightengine_confidence,
                "analyzed_frames": genconvit_result.analyzed_frames if genconvit_result else 0,
                "frame_scores": genconvit_result.frame_scores if genconvit_result else [],
                "model": "GenConViT + Sightengine Ensemble",
            }
        )

    async def _run_genconvit(self, video_data: bytes):
        """GenConViT 비동기 분석 (sync → async 변환)"""
        detector = get_genconvit_detector_safe()
        if not detector or not detector.is_available():
            return None
        try:
            logger.info("Using GenConViT for video deepfake detection")
            result = await asyncio.to_thread(detector.analyze_video, video_data)
            logger.info(f"GenConViT result: {result.confidence:.1f}%")
            return result
        except Exception as e:
            logger.warning(f"GenConViT failed: {e}")
            return None

    async def _run_sightengine_video(self, video_data: bytes):
        """Sightengine 비디오 비동기 분석"""
        if not self.sightengine.is_configured():
            return None
        try:
            logger.info("Using Sightengine API for video analysis")
            return await self.sightengine.analyze_video(video_data)
        except Exception as e:
            logger.warning(f"Sightengine video analysis failed: {e}")
            return None

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
