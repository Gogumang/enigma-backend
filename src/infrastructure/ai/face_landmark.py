"""
얼굴 랜드마크 감지 서비스 (MediaPipe 기반)
눈, 코, 입 등의 정확한 좌표를 퍼센트로 반환
"""

import io
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# MediaPipe 얼굴 메시 랜드마크 인덱스
# https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
LANDMARKS = {
    "left_eye": [33, 133, 160, 159, 158, 144, 145, 153],  # 왼쪽 눈 주변
    "right_eye": [362, 263, 387, 386, 385, 373, 374, 380],  # 오른쪽 눈 주변
    "left_eye_center": 468,  # 왼쪽 눈 중심 (iris)
    "right_eye_center": 473,  # 오른쪽 눈 중심 (iris)
    "nose_tip": 1,  # 코끝
    "mouth_center": 13,  # 입 중앙 (윗입술)
    "mouth_top": 0,  # 입술 위
    "mouth_bottom": 17,  # 입술 아래
    "left_cheek": 50,  # 왼쪽 볼
    "right_cheek": 280,  # 오른쪽 볼
    "forehead": 10,  # 이마
    "chin": 152,  # 턱
}


@dataclass
class FaceLandmark:
    """얼굴 랜드마크 좌표"""
    label: str
    x: float  # 0-100 퍼센트
    y: float  # 0-100 퍼센트
    description: str = ""


@dataclass
class FaceLandmarkResult:
    """얼굴 랜드마크 감지 결과"""
    success: bool
    landmarks: list[FaceLandmark]
    face_detected: bool = False
    error: str | None = None


class FaceLandmarkService:
    """얼굴 랜드마크 감지 서비스"""

    def __init__(self):
        self._face_mesh = None
        self._initialized = False

    def _ensure_initialized(self):
        """MediaPipe 초기화 (지연 로딩)"""
        if self._initialized:
            return

        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,  # iris 랜드마크 포함
                min_detection_confidence=0.5,
            )
            self._initialized = True
            logger.info("MediaPipe FaceMesh initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise

    def detect_landmarks(self, image_data: bytes) -> FaceLandmarkResult:
        """이미지에서 얼굴 랜드마크 감지"""
        try:
            self._ensure_initialized()

            # 이미지 로드
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image_array = np.array(image)
            height, width = image_array.shape[:2]

            # MediaPipe로 얼굴 메시 감지
            results = self._face_mesh.process(image_array)

            if not results.multi_face_landmarks:
                return FaceLandmarkResult(
                    success=False,
                    landmarks=[],
                    face_detected=False,
                    error="얼굴을 찾을 수 없습니다"
                )

            # 첫 번째 얼굴의 랜드마크
            face_landmarks = results.multi_face_landmarks[0]

            # 주요 랜드마크 좌표 추출
            landmarks = []

            # 왼쪽 눈 (이미지 기준 왼쪽 = 사람 기준 오른쪽 눈)
            left_eye = self._get_landmark_center(
                face_landmarks, LANDMARKS["left_eye"], width, height
            )
            if left_eye:
                landmarks.append(FaceLandmark(
                    label="왼쪽 눈",
                    x=left_eye[0],
                    y=left_eye[1],
                ))

            # 오른쪽 눈
            right_eye = self._get_landmark_center(
                face_landmarks, LANDMARKS["right_eye"], width, height
            )
            if right_eye:
                landmarks.append(FaceLandmark(
                    label="오른쪽 눈",
                    x=right_eye[0],
                    y=right_eye[1],
                ))

            # 코
            nose = self._get_single_landmark(
                face_landmarks, LANDMARKS["nose_tip"], width, height
            )
            if nose:
                landmarks.append(FaceLandmark(
                    label="코",
                    x=nose[0],
                    y=nose[1],
                ))

            # 입
            mouth_top = self._get_single_landmark(
                face_landmarks, LANDMARKS["mouth_top"], width, height
            )
            mouth_bottom = self._get_single_landmark(
                face_landmarks, LANDMARKS["mouth_bottom"], width, height
            )
            if mouth_top and mouth_bottom:
                mouth_x = (mouth_top[0] + mouth_bottom[0]) / 2
                mouth_y = (mouth_top[1] + mouth_bottom[1]) / 2
                landmarks.append(FaceLandmark(
                    label="입",
                    x=mouth_x,
                    y=mouth_y,
                ))

            # 왼쪽 볼
            left_cheek = self._get_single_landmark(
                face_landmarks, LANDMARKS["left_cheek"], width, height
            )
            if left_cheek:
                landmarks.append(FaceLandmark(
                    label="왼쪽 볼",
                    x=left_cheek[0],
                    y=left_cheek[1],
                ))

            # 오른쪽 볼
            right_cheek = self._get_single_landmark(
                face_landmarks, LANDMARKS["right_cheek"], width, height
            )
            if right_cheek:
                landmarks.append(FaceLandmark(
                    label="오른쪽 볼",
                    x=right_cheek[0],
                    y=right_cheek[1],
                ))

            # 이마
            forehead = self._get_single_landmark(
                face_landmarks, LANDMARKS["forehead"], width, height
            )
            if forehead:
                landmarks.append(FaceLandmark(
                    label="이마",
                    x=forehead[0],
                    y=forehead[1],
                ))

            # 턱
            chin = self._get_single_landmark(
                face_landmarks, LANDMARKS["chin"], width, height
            )
            if chin:
                landmarks.append(FaceLandmark(
                    label="턱",
                    x=chin[0],
                    y=chin[1],
                ))

            return FaceLandmarkResult(
                success=True,
                landmarks=landmarks,
                face_detected=True,
            )

        except Exception as e:
            logger.error(f"Face landmark detection failed: {e}")
            return FaceLandmarkResult(
                success=False,
                landmarks=[],
                face_detected=False,
                error=str(e)
            )

    def _get_landmark_center(
        self,
        face_landmarks,
        indices: list[int],
        width: int,
        height: int
    ) -> tuple[float, float] | None:
        """여러 랜드마크의 중심점 계산"""
        try:
            x_sum = 0
            y_sum = 0
            for idx in indices:
                landmark = face_landmarks.landmark[idx]
                x_sum += landmark.x
                y_sum += landmark.y

            x_percent = (x_sum / len(indices)) * 100
            y_percent = (y_sum / len(indices)) * 100

            return (round(x_percent, 1), round(y_percent, 1))
        except Exception:
            return None

    def _get_single_landmark(
        self,
        face_landmarks,
        index: int,
        width: int,
        height: int
    ) -> tuple[float, float] | None:
        """단일 랜드마크 좌표"""
        try:
            landmark = face_landmarks.landmark[index]
            x_percent = landmark.x * 100
            y_percent = landmark.y * 100
            return (round(x_percent, 1), round(y_percent, 1))
        except Exception:
            return None

    def get_analysis_markers(
        self,
        image_data: bytes,
        is_deepfake: bool = True,
        count: int = 3
    ) -> list[dict]:
        """딥페이크 분석용 마커 생성 (실제 좌표 기반)"""
        result = self.detect_landmarks(image_data)

        if not result.success or not result.landmarks:
            # 얼굴 감지 실패 시 기본 좌표 반환
            return self._get_default_markers(is_deepfake)

        # 분석에 사용할 주요 부위 선택 (눈, 피부/볼, 입 or 코)
        priority_labels = ["왼쪽 눈", "오른쪽 눈", "왼쪽 볼", "오른쪽 볼", "입", "코", "이마"]
        selected = []

        for label in priority_labels:
            if len(selected) >= count:
                break
            for lm in result.landmarks:
                if lm.label == label and lm not in selected:
                    selected.append(lm)
                    break

        # 마커 생성
        markers = []
        for i, lm in enumerate(selected):
            markers.append({
                "id": i + 1,
                "x": lm.x,
                "y": lm.y,
                "label": lm.label,
                "description": ""  # AI가 채울 부분
            })

        return markers

    def _get_default_markers(self, is_deepfake: bool) -> list[dict]:
        """얼굴 감지 실패 시 기본 마커"""
        if is_deepfake:
            return [
                {"id": 1, "x": 38, "y": 42, "label": "왼쪽 눈", "description": ""},
                {"id": 2, "x": 62, "y": 42, "label": "오른쪽 눈", "description": ""},
                {"id": 3, "x": 50, "y": 58, "label": "피부", "description": ""},
            ]
        else:
            return [
                {"id": 1, "x": 38, "y": 42, "label": "왼쪽 눈", "description": ""},
                {"id": 2, "x": 62, "y": 42, "label": "오른쪽 눈", "description": ""},
                {"id": 3, "x": 50, "y": 55, "label": "피부", "description": ""},
            ]


# 싱글톤 인스턴스
_face_landmark_service: FaceLandmarkService | None = None


def get_face_landmark_service() -> FaceLandmarkService:
    """FaceLandmarkService 싱글톤 반환"""
    global _face_landmark_service
    if _face_landmark_service is None:
        _face_landmark_service = FaceLandmarkService()
    return _face_landmark_service
