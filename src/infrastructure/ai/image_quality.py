"""이미지 품질 분석 서비스"""

import io
import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


class ImageQualityLevel(Enum):
    """이미지 품질 수준"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ImageQualityResult:
    """이미지 품질 분석 결과"""
    quality_level: ImageQualityLevel
    blur_score: float  # 0-100 (높을수록 선명)
    is_reliable: bool  # 분석 신뢰도가 충분한지
    warning_message: str | None = None


class ImageQualityService:
    """이미지 품질 분석 서비스 (Blur Detection)"""

    # Blur score 임계값
    HIGH_QUALITY_THRESHOLD = 50.0
    MEDIUM_QUALITY_THRESHOLD = 25.0

    def analyze(self, image_data: bytes) -> ImageQualityResult:
        """
        이미지 품질 분석 (흐림 정도 측정)

        Laplacian variance 기반 blur detection:
        - Edge detection 후 분산 계산
        - 분산이 낮으면 흐린 이미지
        """
        try:
            image = Image.open(io.BytesIO(image_data))

            # Grayscale 변환
            if image.mode != "L":
                gray = image.convert("L")
            else:
                gray = image

            # Edge detection (Laplacian approximation)
            edges = gray.filter(ImageFilter.FIND_EDGES)

            # numpy array로 변환
            edge_array = np.array(edges, dtype=np.float64)

            # 분산 계산 (blur score)
            variance = edge_array.var()

            # 0-100 스케일로 정규화 (경험적 값 기반)
            # 일반적으로 variance 100 이상이면 선명한 이미지
            blur_score = min(100.0, (variance / 100.0) * 100.0)

            # 품질 레벨 결정
            if blur_score >= self.HIGH_QUALITY_THRESHOLD:
                quality_level = ImageQualityLevel.HIGH
                is_reliable = True
                warning = None
            elif blur_score >= self.MEDIUM_QUALITY_THRESHOLD:
                quality_level = ImageQualityLevel.MEDIUM
                is_reliable = True
                warning = "이미지 품질이 중간 수준입니다. 분석 정확도가 다소 낮을 수 있습니다."
            else:
                quality_level = ImageQualityLevel.LOW
                is_reliable = False
                warning = "이미지가 흐리거나 저화질입니다. 분석 결과의 신뢰도가 낮습니다. 더 선명한 이미지를 사용하시면 정확한 분석이 가능합니다."

            logger.info(
                f"Image quality analysis: blur_score={blur_score:.2f}, "
                f"quality={quality_level.value}, reliable={is_reliable}"
            )

            return ImageQualityResult(
                quality_level=quality_level,
                blur_score=round(blur_score, 2),
                is_reliable=is_reliable,
                warning_message=warning
            )

        except Exception as e:
            logger.error(f"Image quality analysis failed: {e}")
            # 분석 실패 시 기본값 반환 (분석은 계속 진행)
            return ImageQualityResult(
                quality_level=ImageQualityLevel.MEDIUM,
                blur_score=50.0,
                is_reliable=True,
                warning_message=None
            )
