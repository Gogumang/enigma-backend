"""
SigLIP 기반 AI 생성 이미지 탐지
Google SigLIP-L/14 모델을 활용한 AI 생성 이미지/딥페이크 감지
"""
import io
import logging
from dataclasses import dataclass

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class CLIPDetectionResult:
    """CLIP/SigLIP 탐지 결과"""
    is_ai_generated: bool
    confidence: float  # 0-100
    real_score: float
    fake_score: float
    details: dict


class SigLIPDeepfakeDetector:
    """SigLIP-L/14 기반 딥페이크/AI 생성 이미지 탐지"""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None
        self._initialized = False

    def _ensure_initialized(self):
        """모델 초기화 (지연 로딩)"""
        if self._initialized:
            return

        try:
            from transformers import AutoModel, AutoProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"SigLIP Detector using device: {self._device}")

            self._model = AutoModel.from_pretrained(
                "google/siglip-large-patch16-384"
            ).eval().to(self._device)
            self._processor = AutoProcessor.from_pretrained(
                "google/siglip-large-patch16-384"
            )

            self._initialized = True
            logger.info("SigLIP Deepfake Detector initialized successfully")

        except ImportError:
            logger.warning(
                "transformers not installed. Run: pip install transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize SigLIP detector: {e}")
            raise

    def analyze(self, image_data: bytes) -> CLIPDetectionResult:
        """이미지 분석"""
        self._ensure_initialized()

        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")

            texts = [
                "a real authentic photograph",
                "a fake AI-generated deepfake image",
            ]

            inputs = self._processor(
                text=texts, images=image, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                # SigLIP uses sigmoid instead of softmax
                probs = torch.sigmoid(outputs.logits_per_image[0])

            p_real = probs[0].item()
            p_fake = probs[1].item()

            # 정규화
            total = p_real + p_fake
            if total > 0:
                real_prob = p_real / total
                fake_prob = p_fake / total
            else:
                real_prob = 0.5
                fake_prob = 0.5

            is_ai_generated = fake_prob > 0.5
            confidence = fake_prob * 100 if is_ai_generated else real_prob * 100

            return CLIPDetectionResult(
                is_ai_generated=is_ai_generated,
                confidence=confidence,
                real_score=real_prob * 100,
                fake_score=fake_prob * 100,
                details={
                    "raw_real_score": p_real,
                    "raw_fake_score": p_fake,
                    "real_probability": real_prob * 100,
                    "fake_probability": fake_prob * 100,
                },
            )

        except Exception as e:
            logger.error(f"SigLIP analysis failed: {e}")
            return CLIPDetectionResult(
                is_ai_generated=False,
                confidence=50.0,
                real_score=50.0,
                fake_score=50.0,
                details={"error": str(e)},
            )

    def is_available(self) -> bool:
        """서비스 사용 가능 여부"""
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False


# 싱글톤
_clip_detector: SigLIPDeepfakeDetector | None = None


def get_clip_detector() -> SigLIPDeepfakeDetector:
    """SigLIP Detector 싱글톤"""
    global _clip_detector
    if _clip_detector is None:
        _clip_detector = SigLIPDeepfakeDetector()
    return _clip_detector
