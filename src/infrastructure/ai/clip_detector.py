"""
CLIP 기반 AI 생성 이미지 탐지
OpenAI CLIP 모델을 활용한 AI 생성 이미지/딥페이크 감지
"""
import io
import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class CLIPDetectionResult:
    """CLIP 탐지 결과"""
    is_ai_generated: bool
    confidence: float  # 0-100
    real_score: float
    fake_score: float
    details: dict


class CLIPDeepfakeDetector:
    """CLIP 기반 딥페이크/AI 생성 이미지 탐지"""

    def __init__(self):
        self._model = None
        self._preprocess = None
        self._device = None
        self._initialized = False
        # 탐지용 텍스트 프롬프트
        self._prompts = {
            "real": [
                "a real photograph",
                "an authentic photo",
                "a genuine photograph of a person",
                "a real camera photo",
                "an unedited photograph",
            ],
            "fake": [
                "an AI generated image",
                "a deepfake image",
                "a synthetic image",
                "a computer generated face",
                "an artificially created image",
                "a GAN generated image",
                "a fake photograph",
                "a digitally manipulated image",
            ]
        }

    def _ensure_initialized(self):
        """모델 초기화 (지연 로딩)"""
        if self._initialized:
            return

        try:
            # SSL 인증서 검증 우회 (기업 네트워크 등에서 필요)
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context

            import clip

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"CLIP Detector using device: {self._device}")

            # CLIP 모델 로드 (ViT-B/32 사용)
            self._model, self._preprocess = clip.load("ViT-B/32", device=self._device)

            # 텍스트 임베딩 미리 계산
            self._real_text_features = self._encode_texts(self._prompts["real"])
            self._fake_text_features = self._encode_texts(self._prompts["fake"])

            self._initialized = True
            logger.info("CLIP Deepfake Detector initialized successfully")

        except ImportError:
            logger.warning("CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize CLIP detector: {e}")
            raise

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """텍스트를 CLIP 임베딩으로 변환"""
        import clip
        text_tokens = clip.tokenize(texts).to(self._device)
        with torch.no_grad():
            text_features = self._model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def analyze(self, image_data: bytes) -> CLIPDetectionResult:
        """이미지 분석"""
        self._ensure_initialized()

        try:
            # 이미지 로드
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 전처리
            image_input = self._preprocess(image).unsqueeze(0).to(self._device)

            # 이미지 임베딩 계산
            with torch.no_grad():
                image_features = self._model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Real 프롬프트들과의 유사도
            real_similarities = (image_features @ self._real_text_features.T).squeeze()
            real_score = real_similarities.mean().item()

            # Fake 프롬프트들과의 유사도
            fake_similarities = (image_features @ self._fake_text_features.T).squeeze()
            fake_score = fake_similarities.mean().item()

            # Softmax로 확률 계산
            scores = torch.tensor([real_score, fake_score])
            probs = torch.softmax(scores * 50, dim=0)  # temperature scaling

            real_prob = probs[0].item()
            fake_prob = probs[1].item()

            # AI 생성 여부 판정
            is_ai_generated = fake_prob > 0.5
            confidence = fake_prob * 100 if is_ai_generated else (1 - fake_prob) * 100

            return CLIPDetectionResult(
                is_ai_generated=is_ai_generated,
                confidence=confidence,
                real_score=real_prob * 100,
                fake_score=fake_prob * 100,
                details={
                    "raw_real_score": real_score,
                    "raw_fake_score": fake_score,
                    "real_probability": real_prob * 100,
                    "fake_probability": fake_prob * 100,
                }
            )

        except Exception as e:
            logger.error(f"CLIP analysis failed: {e}")
            # 실패 시 중립 결과 반환
            return CLIPDetectionResult(
                is_ai_generated=False,
                confidence=50.0,
                real_score=50.0,
                fake_score=50.0,
                details={"error": str(e)}
            )

    def is_available(self) -> bool:
        """서비스 사용 가능 여부"""
        try:
            import clip
            return True
        except ImportError:
            return False


# 싱글톤
_clip_detector: CLIPDeepfakeDetector | None = None


def get_clip_detector() -> CLIPDeepfakeDetector:
    """CLIP Detector 싱글톤"""
    global _clip_detector
    if _clip_detector is None:
        _clip_detector = CLIPDeepfakeDetector()
    return _clip_detector
