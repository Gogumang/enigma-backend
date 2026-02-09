"""
GenConViT 비디오 딥페이크 탐지
ConvNeXt-Tiny + Swin-Tiny + Autoencoder (DFDC+FF+++CelebDF+TIMIT AUC 0.993)
Reference: https://github.com/erprogs/GenConViT
"""
import logging
import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class GenConViTResult:
    """GenConViT 탐지 결과"""
    is_deepfake: bool
    confidence: float  # 0-100
    frame_scores: list[float]
    analyzed_frames: int
    details: dict


# ==================== GenConViT ED 모델 정의 ====================

class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding: Swin → ConvNeXt patch_embed 교체"""
    def __init__(self, backbone, img_size=224, patch_size=1, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = (feature_size, feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Encoder(nn.Module):
    """5-layer Conv Encoder: 3→16→32→64→128→256"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
        )

    def forward(self, x):
        return self.features(x)


class Decoder(nn.Module):
    """5-layer Transposed Conv Decoder: 256→128→64→32→16→3"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 3, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)


def _get_backbone_out_features(backbone) -> int:
    """timm 버전 호환 backbone output features 가져오기"""
    # timm >= 1.0: backbone.head.fc or backbone.head
    if hasattr(backbone, 'head'):
        head = backbone.head
        if hasattr(head, 'fc') and hasattr(head.fc, 'out_features'):
            return head.fc.out_features
        if hasattr(head, 'out_features'):
            return head.out_features
    # timm < 1.0: backbone.num_classes or backbone.get_classifier()
    if hasattr(backbone, 'num_classes'):
        return backbone.num_classes
    return 1000  # ImageNet default


class GenConViTED(nn.Module):
    """
    GenConViT ED (Encoder-Decoder) for Deepfake Detection
    encode → decode → backbone(decoded) + backbone(original) → concat → FC → 2 classes
    """
    def __init__(self, pretrained=True):
        super().__init__()
        import timm

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.backbone = timm.create_model('convnext_tiny', pretrained=pretrained)
        self.embedder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
        self.backbone.patch_embed = HybridEmbed(self.embedder, img_size=224, embed_dim=768)

        num_features = _get_backbone_out_features(self.backbone) * 2
        self.fc = nn.Linear(num_features, num_features // 4)
        self.fc2 = nn.Linear(num_features // 4, 2)
        self.relu = nn.GELU()

    def forward(self, images):
        encimg = self.encoder(images)
        decimg = self.decoder(encimg)

        x1 = self.backbone(decimg)
        x2 = self.backbone(images)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc2(self.relu(self.fc(self.relu(x))))
        return x


# ==================== GenConViT 서비스 ====================

class GenConViTDetector:
    """GenConViT 비디오 딥페이크 탐지 서비스"""

    def __init__(self):
        self._model: GenConViTED | None = None
        self._device = None
        self._initialized = False
        self._image_size = 224
        self._num_frames = 15  # GenConViT 기본값

    def _ensure_initialized(self):
        """모델 초기화 + HuggingFace 가중치 자동 다운로드"""
        if self._initialized:
            return

        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"GenConViT using device: {self._device}")

            # 모델 생성 (pretrained=False: 가중치는 별도 로드)
            self._model = GenConViTED(pretrained=False)

            # HuggingFace에서 사전 훈련 가중치 다운로드
            from huggingface_hub import hf_hub_download
            weight_path = hf_hub_download(
                repo_id="Deressa/GenConViT",
                filename="genconvit_ed_inference.pth",
            )
            state_dict = torch.load(weight_path, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state_dict)
            logger.info(f"Loaded GenConViT ED weights from HuggingFace ({weight_path})")

            self._model.eval()
            self._model = self._model.to(self._device)
            self._initialized = True
            logger.info("GenConViT initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize GenConViT: {e}")
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

                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)

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

    def _extract_frames(self, video_data: bytes, max_frames: int = 15) -> list[np.ndarray]:
        """비디오에서 프레임 추출 (얼굴 크롭 포함)"""
        raw_frames = []

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(video_data)
            temp_path = f.name

        try:
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                raise ValueError("Cannot read video frames")

            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    raw_frames.append(frame)

            cap.release()

        finally:
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
        """프레임 전처리 → 4D 텐서 (frames, C, H, W)"""
        processed = []
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        for frame in frames:
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - mean) / std
            frame = np.transpose(frame, (2, 0, 1))  # HWC → CHW
            processed.append(frame)

        # (frames, 3, 224, 224) — 4D, no batch dim
        return torch.tensor(np.stack(processed), dtype=torch.float32)

    def analyze_video(self, video_data: bytes) -> GenConViTResult:
        """비디오 분석 (프레임별 추론 후 평균 집계)"""
        self._ensure_initialized()

        try:
            frames = self._extract_frames(video_data, self._num_frames)

            if len(frames) < 5:
                raise ValueError(f"Not enough frames extracted: {len(frames)}")

            logger.info(f"Extracted {len(frames)} frames for analysis")

            input_tensor = self._preprocess_frames(frames).to(self._device)

            # 프레임별 추론: (frames, 3, 224, 224) → (frames, 2)
            with torch.no_grad():
                logits = self._model(input_tensor)
                probs = torch.sigmoid(logits)  # (frames, 2)

            # 프레임 평균 → 최종 예측
            mean_probs = torch.mean(probs, dim=0)  # (2,)

            # GenConViT XOR 라벨 매핑: class 0 = FAKE, class 1 = REAL
            # fake 확률 = probs[:, 0]
            fake_prob = mean_probs[0].item()
            prediction = torch.argmax(mean_probs).item()
            # XOR: argmax=0 → 0^1=1 → FAKE, argmax=1 → 1^1=0 → REAL
            is_deepfake = (prediction ^ 1) == 1  # True if FAKE

            confidence = fake_prob * 100

            # 프레임별 fake 확률
            frame_scores = (probs[:, 0] * 100).tolist()

            return GenConViTResult(
                is_deepfake=is_deepfake,
                confidence=confidence,
                frame_scores=frame_scores,
                analyzed_frames=len(frames),
                details={
                    "model": "GenConViT-ED",
                    "auc": 0.993,
                    "mean_probs": mean_probs.tolist(),
                }
            )

        except Exception as e:
            logger.error(f"GenConViT video analysis failed: {e}")
            return GenConViTResult(
                is_deepfake=False,
                confidence=50.0,
                frame_scores=[],
                analyzed_frames=0,
                details={"error": str(e)}
            )

    def is_available(self) -> bool:
        """서비스 사용 가능 여부"""
        try:
            import timm  # noqa: F401
            return True
        except ImportError:
            return False


# 싱글톤
_genconvit_detector: GenConViTDetector | None = None


def get_genconvit_detector() -> GenConViTDetector:
    """GenConViT Detector 싱글톤"""
    global _genconvit_detector
    if _genconvit_detector is None:
        _genconvit_detector = GenConViTDetector()
    return _genconvit_detector
