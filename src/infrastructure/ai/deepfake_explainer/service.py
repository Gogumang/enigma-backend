"""
Explainable Deepfake Detection Service
EfficientViT + Attention Heatmap
"""
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from albumentations import Compose, PadIfNeeded
from PIL import Image
from torch.nn import AvgPool2d

from .transforms import IsotropicResize

logger = logging.getLogger(__name__)

# 모델 파일 경로
MODEL_DIR = Path(__file__).parent / "models"
CONFIG_PATH = Path(__file__).parent / "explained_architecture.yaml"


@dataclass
class DetectionMarker:
    """탐지된 영역 마커"""
    id: int
    x: float  # 0-100 퍼센트
    y: float  # 0-100 퍼센트
    intensity: float  # 0-1 강도
    label: str
    description: str


@dataclass
class ExplainerResult:
    """딥페이크 탐지 결과"""
    is_deepfake: bool
    confidence: float  # 0-100
    markers: list[DetectionMarker]
    heatmap_base64: str | None = None  # 히트맵 이미지 (base64)
    raw_heatmap: np.ndarray | None = None  # 히트맵 배열


def avg_heads(cam, grad):
    """Rule 5: Average attention heads"""
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


def apply_self_attention_rules(R_ss, cam_ss):
    """Rule 6: Apply self-attention rules"""
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


class DeepfakeExplainerService:
    """딥페이크 탐지 및 히트맵 생성 서비스"""

    def __init__(self):
        self._model = None
        self._config = None
        self._device = None
        self._initialized = False
        self._down_sample = None

    def _ensure_initialized(self):
        """모델 초기화 (지연 로딩)"""
        if self._initialized:
            return

        try:
            # GPU/CPU 설정
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self._device}")

            # Config 로드
            with open(CONFIG_PATH, 'r') as f:
                self._config = yaml.safe_load(f)

            # 모델 로드
            model_path = MODEL_DIR / "efficientnetB0_checkpoint89_All"

            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                logger.warning("Download from: https://drive.google.com/drive/folders/1-JtWGMyd7YaTa56R6uYpjvwmUyW5q-zN")
                raise FileNotFoundError(f"Model not found: {model_path}")

            from .evit_model import EfficientViT

            self._model = EfficientViT(
                config=self._config,
                channels=1280,
                selected_efficient_net=0
            )
            self._model.load_state_dict(torch.load(model_path, map_location=self._device))
            self._model.eval()
            self._model = self._model.to(self._device)

            self._down_sample = AvgPool2d(kernel_size=2)
            self._initialized = True
            logger.info("DeepfakeExplainerService initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize DeepfakeExplainerService: {e}")
            raise

    def _create_transform(self, size: int):
        """이미지 전처리 transform 생성"""
        return Compose([
            IsotropicResize(
                max_side=size,
                interpolation_down=cv2.INTER_AREA,
                interpolation_up=cv2.INTER_CUBIC
            ),
            PadIfNeeded(
                min_height=size,
                min_width=size,
                border_mode=cv2.BORDER_CONSTANT
            ),
        ])

    def _generate_relevance(self, input_tensor):
        """Attention relevance map 생성"""
        output = self._model(input_tensor, register_hook=True)
        self._model.zero_grad()
        output.backward(retain_graph=True)

        num_tokens = self._model.transformer.blocks[0].attn.get_attention_map().shape[-1]
        R = torch.eye(num_tokens, num_tokens, device=self._device)

        for blk in self._model.transformer.blocks:
            grad = blk.attn.get_attn_gradients()
            cam = blk.attn.get_attention_map()
            cam = avg_heads(cam, grad)
            R += apply_self_attention_rules(R, cam.to(self._device))

        return R[0, 1:]

    def _create_heatmap(self, original_image: torch.Tensor) -> np.ndarray:
        """히트맵 생성"""
        transformer_attribution = self._generate_relevance(
            original_image.unsqueeze(0).to(self._device)
        ).detach()

        transformer_attribution = transformer_attribution.reshape(1, 1, 32, 32)
        transformer_attribution = self._down_sample(transformer_attribution)
        transformer_attribution = torch.nn.functional.interpolate(
            transformer_attribution,
            scale_factor=14,
            mode='bilinear'
        )
        transformer_attribution = transformer_attribution.reshape(224, 224)
        transformer_attribution = transformer_attribution.cpu().numpy()

        # 정규화
        attr_min = transformer_attribution.min()
        attr_max = transformer_attribution.max()
        if attr_max > attr_min:
            transformer_attribution = (transformer_attribution - attr_min) / (attr_max - attr_min)
        else:
            transformer_attribution = np.zeros_like(transformer_attribution)

        return transformer_attribution

    def _overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """이미지에 히트맵 오버레이"""
        # 히트맵을 컬러맵으로 변환
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = np.float32(heatmap_colored) / 255

        # 이미지 정규화
        image_norm = np.float32(image) / 255 if image.max() > 1 else np.float32(image)

        # 오버레이
        cam = heatmap_colored + image_norm
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)

        return cam

    def _extract_markers_from_heatmap(
        self,
        heatmap: np.ndarray,
        is_deepfake: bool,
        top_k: int = 3
    ) -> list[DetectionMarker]:
        """히트맵에서 주요 영역 마커 추출"""
        markers = []

        # 히트맵 블러링으로 노이즈 제거
        heatmap_blur = cv2.GaussianBlur(heatmap, (15, 15), 0)

        # 상위 k개 지점 찾기
        flat_idx = np.argsort(heatmap_blur.flatten())[::-1]
        h, w = heatmap_blur.shape

        used_positions = []
        min_distance = 30  # 최소 거리 (픽셀)

        for idx in flat_idx:
            if len(markers) >= top_k:
                break

            y, x = divmod(idx, w)
            intensity = heatmap_blur[y, x]

            # 이미 가까운 위치가 있으면 스킵
            too_close = False
            for ux, uy in used_positions:
                if np.sqrt((x - ux)**2 + (y - uy)**2) < min_distance:
                    too_close = True
                    break

            if too_close:
                continue

            used_positions.append((x, y))

            # 퍼센트 좌표로 변환
            x_pct = (x / w) * 100
            y_pct = (y / h) * 100

            # 위치에 따른 라벨 결정
            label = self._get_region_label(x_pct, y_pct)

            if is_deepfake:
                description = f"AI 생성 흔적이 감지된 영역 (강도: {intensity:.2f})"
            else:
                description = f"분석된 영역 (강도: {intensity:.2f})"

            markers.append(DetectionMarker(
                id=len(markers) + 1,
                x=round(x_pct, 1),
                y=round(y_pct, 1),
                intensity=float(intensity),
                label=label,
                description=description
            ))

        return markers

    def _get_region_label(self, x: float, y: float) -> str:
        """좌표에 따른 영역 라벨"""
        # 대략적인 얼굴 영역 매핑
        if y < 30:
            return "이마/헤어라인"
        elif y < 50:
            if x < 40:
                return "왼쪽 눈"
            elif x > 60:
                return "오른쪽 눈"
            else:
                return "미간/코"
        elif y < 65:
            if x < 35:
                return "왼쪽 볼"
            elif x > 65:
                return "오른쪽 볼"
            else:
                return "코/입 주변"
        elif y < 80:
            return "입/턱"
        else:
            return "턱/목"

    def analyze(self, image_data: bytes) -> ExplainerResult:
        """이미지 분석 및 히트맵 생성"""
        self._ensure_initialized()

        try:
            # 이미지 로드
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image_array = np.array(image)

            # Transform 적용
            image_size = self._config['model']['image-size']
            transform = self._create_transform(image_size)
            transformed = transform(image=image_array)['image']

            # Tensor 변환
            t_image = torch.tensor(np.asarray(transformed))
            t_image = np.transpose(t_image, (2, 0, 1)).float()

            # 예측
            with torch.no_grad():
                pred_score = torch.sigmoid(
                    self._model(t_image.unsqueeze(0).to(self._device))
                )

            confidence = pred_score.item() * 100
            is_deepfake = confidence > 55  # 55% 이상이면 딥페이크로 판정

            # 히트맵 생성
            heatmap = self._create_heatmap(t_image)

            # 마커 추출
            markers = self._extract_markers_from_heatmap(heatmap, is_deepfake, top_k=3)

            # 히트맵 오버레이 이미지 생성
            overlay = self._overlay_heatmap(transformed, heatmap)

            # Base64 인코딩
            import base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

            return ExplainerResult(
                is_deepfake=is_deepfake,
                confidence=confidence,
                markers=markers,
                heatmap_base64=heatmap_base64,
                raw_heatmap=heatmap
            )

        except Exception as e:
            logger.error(f"Deepfake analysis failed: {e}")
            raise

    def is_available(self) -> bool:
        """서비스 사용 가능 여부"""
        model_path = MODEL_DIR / "efficientnetB0_checkpoint89_All"
        return model_path.exists()


# 싱글톤
_service: DeepfakeExplainerService | None = None


def get_deepfake_explainer_service() -> DeepfakeExplainerService:
    """DeepfakeExplainerService 싱글톤"""
    global _service
    if _service is None:
        _service = DeepfakeExplainerService()
    return _service
