"""
Cross-EfficientViT 비디오 딥페이크 탐지
EfficientNet + Vision Transformer 결합 (DFDC AUC 0.951)
Reference: https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection
"""
import io
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image

logger = logging.getLogger(__name__)

# 모델 파일 경로
MODEL_DIR = Path(__file__).parent / "deepfake_explainer" / "models"


@dataclass
class CrossEViTResult:
    """Cross-EfficientViT 탐지 결과"""
    is_deepfake: bool
    confidence: float  # 0-100
    frame_scores: list[float]
    analyzed_frames: int
    details: dict


# ==================== Cross-EfficientViT 모델 정의 ====================

class CrossAttention(nn.Module):
    """Cross Attention between EfficientNet features and ViT"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        b, n, _ = x.shape
        h = self.heads

        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim=-1)
        k, v = kv

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossEfficientViT(nn.Module):
    """
    Cross-EfficientViT for Video Deepfake Detection
    Combines EfficientNet features with Vision Transformer using cross-attention
    """
    def __init__(
        self,
        num_classes=1,
        image_size=224,
        num_frames=30,
        dim=512,
        depth=4,
        heads=8,
        dim_head=64,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super().__init__()
        try:
            from efficientnet_pytorch import EfficientNet
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        except Exception:
            logger.warning("EfficientNet not available, using placeholder")
            self.efficient_net = None

        self.num_frames = num_frames
        self.dim = dim

        # EfficientNet 출력을 dim으로 변환
        self.feature_proj = nn.Linear(1280, dim)

        # Temporal positional embedding
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(dim, heads, dim_head, dropout)
            for _ in range(depth)
        ])

        # Feed-forward layers
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, dim),
                nn.Dropout(dropout),
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, frames):
        """
        Args:
            frames: (batch, num_frames, channels, height, width)
        Returns:
            logits: (batch, num_classes)
        """
        b, t, c, h, w = frames.shape

        # Extract features from all frames at once (batched)
        if self.efficient_net is not None:
            flat_frames = rearrange(frames, 'b t c h w -> (b t) c h w')
            flat_feat = self.efficient_net.extract_features(flat_frames)
            flat_feat = flat_feat.mean(dim=[2, 3])  # Global average pooling
            x = rearrange(flat_feat, '(b t) d -> b t d', b=b, t=t)
        else:
            x = frames.mean(dim=[2, 3, 4]).unsqueeze(-1).expand(-1, -1, 1280)

        # Project to dim
        x = self.feature_proj(x)

        # Add temporal positional embedding
        x = x + self.temporal_pos_embedding[:, :t, :]
        x = self.dropout(x)

        # Cross-attention with temporal context
        for cross_attn, ff in zip(self.cross_attention_layers, self.ff_layers):
            # Self-attention as cross-attention with itself
            x = x + cross_attn(x, x)
            x = x + ff(x)

        x = self.norm(x)

        # Global temporal pooling
        x = x.mean(dim=1)

        return self.mlp_head(x)


class CrossEfficientViTDetector:
    """Cross-EfficientViT 비디오 딥페이크 탐지 서비스"""

    def __init__(self):
        self._model = None
        self._device = None
        self._initialized = False
        self._image_size = 224
        self._num_frames = 16  # 추출 프레임 수 (속도 최적화)

    def _ensure_initialized(self):
        """모델 초기화"""
        if self._initialized:
            return

        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Cross-EfficientViT using device: {self._device}")

            # 모델 생성 (num_frames=30: 체크포인트 positional embedding 호환)
            self._model = CrossEfficientViT(
                num_classes=1,
                image_size=self._image_size,
                num_frames=30,
            )

            # 사전 훈련된 가중치 로드 시도
            model_path = MODEL_DIR / "cross_efficient_vit.pth"
            if model_path.exists():
                self._model.load_state_dict(
                    torch.load(model_path, map_location=self._device)
                )
                logger.info(f"Loaded Cross-EfficientViT weights from {model_path}")
            else:
                logger.warning(
                    f"Cross-EfficientViT weights not found at {model_path}. "
                    "Download from: http://datino.isti.cnr.it/efficientvit_deepfake/cross_efficient_vit.pth"
                )

            self._model.eval()
            self._model = self._model.to(self._device)
            self._initialized = True
            logger.info("Cross-EfficientViT initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Cross-EfficientViT: {e}")
            raise

    def _detect_face_bbox(self, sample_frames: list[np.ndarray]) -> tuple[int, int, int, int] | None:
        """샘플 프레임에서 얼굴 바운딩 박스 탐지 (30% 마진 포함)"""
        try:
            import mediapipe as mp
            face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )

            for frame in sample_frames:
                results = face_detection.process(frame)
                if results.detections:
                    det = results.detections[0]
                    bbox = det.location_data.relative_bounding_box
                    h, w = frame.shape[:2]

                    # 상대 좌표 → 절대 좌표
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)

                    # 30% 마진 추가
                    margin_x = int(bw * 0.3)
                    margin_y = int(bh * 0.3)
                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)
                    x2 = min(w, x1 + bw + 2 * margin_x)
                    y2 = min(h, y1 + bh + 2 * margin_y)

                    face_detection.close()
                    logger.info(f"Face detected: ({x1},{y1})-({x2},{y2}) in {w}x{h} frame")
                    return (x1, y1, x2, y2)

            face_detection.close()
        except Exception as e:
            logger.warning(f"Face detection failed, using full frame: {e}")

        return None

    def _extract_frames(self, video_data: bytes, max_frames: int = 16) -> list[np.ndarray]:
        """비디오에서 프레임 추출 (얼굴 크롭 포함)"""
        raw_frames = []

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(video_data)
            temp_path = f.name

        try:
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                raise ValueError("Cannot read video frames")

            # 균등하게 프레임 샘플링
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    raw_frames.append(frame)

            cap.release()

        finally:
            import os
            os.unlink(temp_path)

        if not raw_frames:
            return []

        # 샘플 3개로 얼굴 탐지
        sample_indices = np.linspace(0, len(raw_frames) - 1, min(3, len(raw_frames)), dtype=int)
        sample_frames = [raw_frames[i] for i in sample_indices]
        face_bbox = self._detect_face_bbox(sample_frames)

        # 얼굴 크롭 적용 후 리사이즈
        frames = []
        for frame in raw_frames:
            if face_bbox:
                x1, y1, x2, y2 = face_bbox
                frame = frame[y1:y2, x1:x2]
            frame = cv2.resize(frame, (self._image_size, self._image_size))
            frames.append(frame)

        return frames

    def _preprocess_frames(self, frames: list[np.ndarray]) -> torch.Tensor:
        """프레임 전처리"""
        # Normalize and convert to tensor
        processed = []
        for frame in frames:
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            # Normalize with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame = (frame - mean) / std
            # HWC -> CHW
            frame = np.transpose(frame, (2, 0, 1))
            processed.append(frame)

        # Stack: (num_frames, C, H, W)
        tensor = torch.tensor(np.stack(processed), dtype=torch.float32)
        # Add batch dim: (1, num_frames, C, H, W)
        return tensor.unsqueeze(0)

    def analyze_video(self, video_data: bytes) -> CrossEViTResult:
        """비디오 분석"""
        self._ensure_initialized()

        try:
            # 프레임 추출
            frames = self._extract_frames(video_data, self._num_frames)

            if len(frames) < 5:
                raise ValueError(f"Not enough frames extracted: {len(frames)}")

            logger.info(f"Extracted {len(frames)} frames for analysis")

            # 전처리
            input_tensor = self._preprocess_frames(frames).to(self._device)

            # 예측
            with torch.no_grad():
                logits = self._model(input_tensor)
                prob = torch.sigmoid(logits).item()

            confidence = prob * 100
            is_deepfake = confidence >= 50

            # 전체 신뢰도를 프레임 스코어로 사용
            frame_scores = [confidence] * len(frames)

            return CrossEViTResult(
                is_deepfake=is_deepfake,
                confidence=confidence,
                frame_scores=frame_scores,
                analyzed_frames=len(frames),
                details={
                    "model": "Cross-EfficientViT",
                    "dfdc_auc": 0.951,
                    "raw_logit": logits.item(),
                }
            )

        except Exception as e:
            logger.error(f"Cross-EfficientViT video analysis failed: {e}")
            return CrossEViTResult(
                is_deepfake=False,
                confidence=50.0,
                frame_scores=[],
                analyzed_frames=0,
                details={"error": str(e)}
            )

    def is_available(self) -> bool:
        """서비스 사용 가능 여부"""
        try:
            from efficientnet_pytorch import EfficientNet
            return True
        except ImportError:
            return False


# 싱글톤
_cross_evit_detector: CrossEfficientViTDetector | None = None


def get_cross_evit_detector() -> CrossEfficientViTDetector:
    """Cross-EfficientViT Detector 싱글톤"""
    global _cross_evit_detector
    if _cross_evit_detector is None:
        _cross_evit_detector = CrossEfficientViTDetector()
    return _cross_evit_detector
