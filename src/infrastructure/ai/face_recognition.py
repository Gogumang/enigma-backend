import io
import logging

import numpy as np
from deepface import DeepFace
from PIL import Image

logger = logging.getLogger(__name__)


class FaceRecognitionService:
    """얼굴 인식 서비스 (DeepFace 기반)"""

    def __init__(self, model_name: str = "VGG-Face"):
        self.model_name = model_name
        self._initialized = False

    async def initialize(self) -> None:
        """모델 초기화 (첫 호출 시 자동 다운로드)"""
        try:
            # DeepFace는 첫 호출 시 모델을 자동 다운로드
            logger.info(f"Initializing DeepFace with model: {self.model_name}")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize DeepFace: {e}")
            raise

    def is_ready(self) -> bool:
        return self._initialized

    async def extract_embedding(self, image_data: bytes) -> list[float] | None:
        """이미지에서 얼굴 임베딩 추출"""
        try:
            # 이미지 로드
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)

            # DeepFace로 임베딩 추출
            embeddings = DeepFace.represent(
                img_path=image_array,
                model_name=self.model_name,
                enforce_detection=False
            )

            if embeddings and len(embeddings) > 0:
                return embeddings[0]["embedding"]

            return None

        except Exception as e:
            logger.error(f"Face embedding extraction failed: {e}")
            return None

    async def compare_faces(
        self,
        image_data: bytes,
        target_embedding: list[float]
    ) -> tuple[bool, float]:
        """두 얼굴 비교 (이미지 vs 임베딩)"""
        try:
            source_embedding = await self.extract_embedding(image_data)

            if source_embedding is None:
                return False, 1.0

            # 유클리드 거리 계산
            distance = self._euclidean_distance(source_embedding, target_embedding)

            # 거리가 0.6 이하면 같은 사람으로 판단
            is_match = distance < 0.6

            return is_match, distance

        except Exception as e:
            logger.error(f"Face comparison failed: {e}")
            return False, 1.0

    def _euclidean_distance(self, emb1: list[float], emb2: list[float]) -> float:
        """유클리드 거리 계산"""
        arr1 = np.array(emb1)
        arr2 = np.array(emb2)
        return float(np.linalg.norm(arr1 - arr2))

    async def detect_face(self, image_data: bytes) -> bool:
        """이미지에서 얼굴 감지"""
        try:
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)

            faces = DeepFace.extract_faces(
                img_path=image_array,
                enforce_detection=False
            )

            return len(faces) > 0 and faces[0].get("confidence", 0) > 0.5

        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return False

    async def analyze_face(self, image_data: bytes) -> dict | None:
        """얼굴 분석 (나이, 성별, 감정 등)"""
        try:
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)

            analysis = DeepFace.analyze(
                img_path=image_array,
                actions=["age", "gender", "emotion"],
                enforce_detection=False
            )

            if analysis and len(analysis) > 0:
                return analysis[0]

            return None

        except Exception as e:
            logger.error(f"Face analysis failed: {e}")
            return None
