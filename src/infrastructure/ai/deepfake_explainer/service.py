"""
Explainable Deepfake Detection Service
GenD-PE (WACV 2026) + Attention Heatmap + Multi-Algorithm Analysis
"""
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import maximum_filter

logger = logging.getLogger(__name__)

# 모델 파일 경로
MODEL_DIR = Path(__file__).parent / "models"


@dataclass
class AlgorithmCheck:
    """알고리즘 검사 결과"""
    name: str
    passed: bool
    score: float  # 0-1 (1이 딥페이크 가능성 높음)
    description: str
    weight: float = 1.0  # 가중치


@dataclass
class DetectionMarker:
    """탐지된 영역 마커"""
    id: int
    x: float  # 0-100 퍼센트
    y: float  # 0-100 퍼센트
    intensity: float  # 0-1 강도
    label: str
    description: str
    algorithm_flags: list[str] = field(default_factory=list)  # 해당 영역에서 감지된 알고리즘 이상


@dataclass
class ExplainerResult:
    """딥페이크 탐지 결과"""
    is_deepfake: bool
    confidence: float  # 0-100
    markers: list[DetectionMarker]
    heatmap_base64: str | None = None  # 히트맵 이미지 (base64)
    raw_heatmap: np.ndarray | None = None  # 히트맵 배열
    algorithm_checks: list[AlgorithmCheck] = field(default_factory=list)  # 알고리즘 검사 결과
    ensemble_details: dict = field(default_factory=dict)  # 앙상블 상세 정보


# ==================== 알고리즘 기반 딥페이크 탐지 함수들 ====================

def analyze_frequency_domain(image: np.ndarray) -> AlgorithmCheck:
    """
    주파수 도메인 분석 (DCT/FFT 아티팩트 탐지)
    GAN 생성 이미지는 특정 주파수 패턴이 있음
    """
    try:
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # DCT 변환
        gray_float = np.float32(gray)
        dct = cv2.dct(gray_float)

        # 고주파 영역 에너지 분석
        h, w = dct.shape
        high_freq_region = dct[h//2:, w//2:]
        low_freq_region = dct[:h//4, :w//4]

        high_energy = np.mean(np.abs(high_freq_region))
        low_energy = np.mean(np.abs(low_freq_region)) + 1e-6

        # GAN 이미지는 고주파/저주파 비율이 비정상적
        ratio = high_energy / low_energy

        # FFT 분석
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # 중심에서 방사형 평균 계산
        center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
        y_grid, x_grid = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
        dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)

        # 주파수 대역별 에너지
        radial_bins = np.linspace(0, min(center_x, center_y), 10)
        radial_energy = []
        for i in range(len(radial_bins) - 1):
            mask = (dist_from_center >= radial_bins[i]) & (dist_from_center < radial_bins[i+1])
            radial_energy.append(np.mean(magnitude[mask]) if np.any(mask) else 0)

        # GAN fingerprint: 특정 주파수 대역에서 비정상적 피크
        radial_energy = np.array(radial_energy)
        energy_std = np.std(radial_energy)
        energy_mean = np.mean(radial_energy) + 1e-6

        # 정규화된 변동 계수
        variation_coeff = energy_std / energy_mean

        # 점수 계산 (비정상적 패턴일수록 높음)
        score = min(1.0, (ratio * 0.3 + variation_coeff * 0.7))

        # 0.4 이상이면 의심
        is_suspicious = score > 0.4

        return AlgorithmCheck(
            name="frequency_analysis",
            passed=not is_suspicious,
            score=score,
            description="주파수 도메인에서 GAN 생성 패턴 감지" if is_suspicious else "주파수 분석 정상",
            weight=1.5  # 중요한 지표
        )
    except Exception as e:
        logger.warning(f"Frequency analysis failed: {e}")
        return AlgorithmCheck(
            name="frequency_analysis",
            passed=True,
            score=0.0,
            description="주파수 분석 실패",
            weight=0.0
        )


def analyze_skin_texture(image: np.ndarray) -> AlgorithmCheck:
    """
    피부 텍스처 일관성 분석
    딥페이크는 피부 텍스처가 불균일하거나 너무 매끄러움
    """
    try:
        # LAB 색공간으로 변환 (피부톤 분석에 적합)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]

        # Gabor 필터로 텍스처 분석
        textures = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel(
                (21, 21), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(l_channel, cv2.CV_32F, kernel)
            textures.append(np.var(filtered))

        # 텍스처 방향별 분산
        texture_variance = np.var(textures)
        texture_mean = np.mean(textures)

        # 로컬 텍스처 일관성 (블록별 분석)
        block_size = 32
        h, w = l_channel.shape
        block_variances = []

        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = l_channel[i:i+block_size, j:j+block_size]
                # 라플라시안 분산 (텍스처 복잡도)
                laplacian = cv2.Laplacian(block, cv2.CV_64F)
                block_variances.append(np.var(laplacian))

        if block_variances:
            # 블록간 텍스처 불일치 감지
            block_std = np.std(block_variances)
            block_mean = np.mean(block_variances) + 1e-6
            inconsistency = block_std / block_mean

            # 너무 균일하거나 (GAN smoothing) 너무 불균일하면 (합성 경계) 의심
            if inconsistency < 0.3:  # 너무 균일
                score = 0.6
                description = "피부 텍스처가 비정상적으로 균일함 (AI 생성 의심)"
            elif inconsistency > 1.5:  # 너무 불균일
                score = 0.7
                description = "피부 텍스처 불일치 감지 (합성 경계 의심)"
            else:
                score = inconsistency * 0.3
                description = "피부 텍스처 정상"
        else:
            score = 0.0
            description = "피부 텍스처 분석 불가"

        return AlgorithmCheck(
            name="skin_texture",
            passed=score < 0.5,
            score=min(1.0, score),
            description=description,
            weight=1.2
        )
    except Exception as e:
        logger.warning(f"Skin texture analysis failed: {e}")
        return AlgorithmCheck(
            name="skin_texture",
            passed=True,
            score=0.0,
            description="피부 텍스처 분석 실패",
            weight=0.0
        )


def analyze_color_consistency(image: np.ndarray) -> AlgorithmCheck:
    """
    색상 일관성 분석
    딥페이크는 조명/색온도 불일치가 있을 수 있음
    """
    try:
        # HSV 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # 이미지를 4분면으로 나누어 색상 분포 비교
        height, width = image.shape[:2]
        quadrants = [
            hsv[:height//2, :width//2],
            hsv[:height//2, width//2:],
            hsv[height//2:, :width//2],
            hsv[height//2:, width//2:],
        ]

        # 각 분면의 평균 색조/채도/명도
        quadrant_stats = []
        for q in quadrants:
            q_h, q_s, q_v = cv2.split(q)
            quadrant_stats.append({
                'h_mean': np.mean(q_h),
                's_mean': np.mean(q_s),
                'v_mean': np.mean(q_v),
            })

        # 분면간 색조 차이
        h_diffs = []
        s_diffs = []
        v_diffs = []
        for i in range(len(quadrant_stats)):
            for j in range(i+1, len(quadrant_stats)):
                h_diffs.append(abs(quadrant_stats[i]['h_mean'] - quadrant_stats[j]['h_mean']))
                s_diffs.append(abs(quadrant_stats[i]['s_mean'] - quadrant_stats[j]['s_mean']))
                v_diffs.append(abs(quadrant_stats[i]['v_mean'] - quadrant_stats[j]['v_mean']))

        # 색조 불일치 점수 (얼굴에서 색조는 보통 일관적이어야 함)
        h_inconsistency = np.mean(h_diffs) / 180.0  # 색조는 0-180
        s_inconsistency = np.mean(s_diffs) / 255.0
        v_inconsistency = np.mean(v_diffs) / 255.0

        # 채도 분포 분석 (딥페이크는 채도가 불균일할 수 있음)
        s_histogram = cv2.calcHist([s], [0], None, [32], [0, 256]).flatten()
        s_histogram = s_histogram / (s_histogram.sum() + 1e-6)
        s_entropy = -np.sum(s_histogram * np.log2(s_histogram + 1e-6))

        # 점수 계산
        score = (h_inconsistency * 0.4 + s_inconsistency * 0.3 + v_inconsistency * 0.3)

        # 색상 불일치가 크면 의심
        is_suspicious = score > 0.15 or h_inconsistency > 0.1

        return AlgorithmCheck(
            name="color_consistency",
            passed=not is_suspicious,
            score=min(1.0, score * 3),  # 스케일 조정
            description="색상/조명 불일치 감지" if is_suspicious else "색상 일관성 정상",
            weight=1.0
        )
    except Exception as e:
        logger.warning(f"Color consistency analysis failed: {e}")
        return AlgorithmCheck(
            name="color_consistency",
            passed=True,
            score=0.0,
            description="색상 분석 실패",
            weight=0.0
        )


def analyze_edge_artifacts(image: np.ndarray) -> AlgorithmCheck:
    """
    경계/엣지 아티팩트 분석
    딥페이크는 얼굴 경계에서 블렌딩 아티팩트가 있을 수 있음
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 다중 스케일 엣지 검출
        edges_fine = cv2.Canny(gray, 50, 150)
        edges_coarse = cv2.Canny(gray, 100, 200)

        # 엣지 연속성 분석
        # 딥페이크는 얼굴 경계에서 엣지가 끊기거나 이중으로 나타남

        # Sobel 그래디언트
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

        # 그래디언트 방향
        gradient_direction = np.arctan2(sobely, sobelx)

        # 방향 일관성 분석 (엣지 주변)
        edge_mask = edges_fine > 0
        if np.any(edge_mask):
            edge_directions = gradient_direction[edge_mask]
            direction_consistency = np.std(edge_directions)

            # 엣지 강도 분포
            edge_magnitudes = gradient_magnitude[edge_mask]
            magnitude_variance = np.var(edge_magnitudes) / (np.mean(edge_magnitudes) + 1e-6)

            # 이중 엣지 감지 (dilate 후 비교)
            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges_fine, kernel, iterations=1)
            double_edge_ratio = np.sum(edges_dilated) / (np.sum(edges_fine) + 1e-6)

            # 점수 계산
            score = (
                min(1.0, direction_consistency / np.pi) * 0.3 +
                min(1.0, magnitude_variance / 100) * 0.3 +
                min(1.0, (double_edge_ratio - 1) / 5) * 0.4
            )
        else:
            score = 0.0

        is_suspicious = score > 0.4

        return AlgorithmCheck(
            name="edge_artifacts",
            passed=not is_suspicious,
            score=min(1.0, score),
            description="경계 아티팩트 감지 (블렌딩 흔적)" if is_suspicious else "경계 분석 정상",
            weight=1.3
        )
    except Exception as e:
        logger.warning(f"Edge artifact analysis failed: {e}")
        return AlgorithmCheck(
            name="edge_artifacts",
            passed=True,
            score=0.0,
            description="경계 분석 실패",
            weight=0.0
        )


def analyze_noise_pattern(image: np.ndarray) -> AlgorithmCheck:
    """
    노이즈 패턴 분석
    진짜 사진은 센서 노이즈가 균일하지만, 딥페이크는 노이즈 패턴이 불일치
    """
    try:
        # 각 채널별 노이즈 추출
        noise_maps = []
        for i in range(3):
            channel = image[:, :, i].astype(np.float32)
            # 중간값 필터로 노이즈 제거한 버전
            denoised = cv2.medianBlur(channel.astype(np.uint8), 5).astype(np.float32)
            noise = channel - denoised
            noise_maps.append(noise)

        # 채널간 노이즈 상관관계
        correlations = []
        for i in range(3):
            for j in range(i+1, 3):
                corr = np.corrcoef(noise_maps[i].flatten(), noise_maps[j].flatten())[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)

        # 진짜 사진은 채널간 노이즈가 상관관계가 높음
        mean_correlation = np.mean(correlations)

        # 노이즈 분포 분석
        all_noise = np.concatenate([n.flatten() for n in noise_maps])
        noise_std = np.std(all_noise)
        noise_kurtosis = np.mean((all_noise - np.mean(all_noise))**4) / (noise_std**4 + 1e-6)

        # 노이즈 공간 일관성
        h, w = image.shape[:2]
        quadrant_noise_levels = []
        for i in range(2):
            for j in range(2):
                q_noise = noise_maps[0][i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
                quadrant_noise_levels.append(np.std(q_noise))

        noise_inconsistency = np.std(quadrant_noise_levels) / (np.mean(quadrant_noise_levels) + 1e-6)

        # 점수 계산
        # 상관관계가 낮거나, 노이즈가 너무 균일하거나 불균일하면 의심
        score = 0.0
        if mean_correlation < 0.5:
            score += 0.3
        if noise_inconsistency > 0.5:
            score += 0.4
        if noise_kurtosis < 2 or noise_kurtosis > 10:
            score += 0.3

        is_suspicious = score > 0.4

        return AlgorithmCheck(
            name="noise_pattern",
            passed=not is_suspicious,
            score=min(1.0, score),
            description="노이즈 패턴 불일치 감지" if is_suspicious else "노이즈 패턴 정상",
            weight=1.1
        )
    except Exception as e:
        logger.warning(f"Noise pattern analysis failed: {e}")
        return AlgorithmCheck(
            name="noise_pattern",
            passed=True,
            score=0.0,
            description="노이즈 분석 실패",
            weight=0.0
        )


def analyze_compression_artifacts(image: np.ndarray) -> AlgorithmCheck:
    """
    압축 아티팩트 분석
    여러번 압축된 이미지나 합성 이미지는 압축 아티팩트 패턴이 다름
    """
    try:
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 8x8 블록 경계에서 DCT 블록 아티팩트 분석
        h, w = gray.shape
        block_size = 8

        # 블록 경계 강도 측정
        horizontal_diff = np.abs(np.diff(gray, axis=1))
        vertical_diff = np.abs(np.diff(gray, axis=0))

        # 8픽셀 간격에서의 차이 (JPEG 블록 경계)
        h_block_edges = horizontal_diff[:, block_size-1::block_size]
        v_block_edges = vertical_diff[block_size-1::block_size, :]

        h_non_block = horizontal_diff[:, np.arange(w-1) % block_size != block_size-1]
        v_non_block = vertical_diff[np.arange(h-1) % block_size != block_size-1, :]

        # 블록 경계 vs 비경계 비율
        h_ratio = np.mean(h_block_edges) / (np.mean(h_non_block) + 1e-6)
        v_ratio = np.mean(v_block_edges) / (np.mean(v_non_block) + 1e-6)

        # 비정상적인 블록 경계 패턴 감지
        # 딥페이크는 원본과 다른 압축 히스토리를 가질 수 있음
        block_ratio = (h_ratio + v_ratio) / 2

        # 이중 압축 감지
        score = 0.0
        if block_ratio > 1.5:  # 블록 경계가 뚜렷함 (이상)
            score = min(1.0, (block_ratio - 1) * 0.5)
        elif block_ratio < 0.8:  # 블록 경계가 너무 안보임 (후처리 의심)
            score = min(1.0, (1 - block_ratio) * 0.5)

        is_suspicious = score > 0.3

        return AlgorithmCheck(
            name="compression_artifacts",
            passed=not is_suspicious,
            score=score,
            description="압축 아티팩트 이상 감지" if is_suspicious else "압축 패턴 정상",
            weight=0.8
        )
    except Exception as e:
        logger.warning(f"Compression artifact analysis failed: {e}")
        return AlgorithmCheck(
            name="compression_artifacts",
            passed=True,
            score=0.0,
            description="압축 분석 실패",
            weight=0.0
        )


def run_all_algorithm_checks(image: np.ndarray) -> list[AlgorithmCheck]:
    """모든 알고리즘 검사 실행"""
    checks = []
    checks.append(analyze_frequency_domain(image))
    checks.append(analyze_skin_texture(image))
    checks.append(analyze_color_consistency(image))
    checks.append(analyze_edge_artifacts(image))
    checks.append(analyze_noise_pattern(image))
    checks.append(analyze_compression_artifacts(image))
    return checks


def calculate_ensemble_score(
    model_confidence: float,
    algorithm_checks: list[AlgorithmCheck]
) -> tuple[float, bool, dict]:
    """
    앙상블 점수 계산
    모델 결과와 알고리즘 검사 결과를 결합
    """
    # 모델 가중치
    model_weight = 2.0  # GenD-PE 모델에 높은 가중치

    # 알고리즘 가중치 합
    total_algo_weight = sum(c.weight for c in algorithm_checks if c.weight > 0)

    # 알고리즘 점수 (가중 평균)
    if total_algo_weight > 0:
        algo_score = sum(c.score * c.weight for c in algorithm_checks) / total_algo_weight
    else:
        algo_score = 0.0

    # 앙상블 점수
    total_weight = model_weight + total_algo_weight
    ensemble_confidence = (
        (model_confidence / 100.0) * model_weight +
        algo_score * total_algo_weight
    ) / total_weight * 100

    # 신뢰도 조정
    # 알고리즘 검사에서 여러개가 의심이면 더 높은 점수
    suspicious_count = sum(1 for c in algorithm_checks if not c.passed)
    if suspicious_count >= 3:
        ensemble_confidence = min(100, ensemble_confidence * 1.2)
    elif suspicious_count >= 2:
        ensemble_confidence = min(100, ensemble_confidence * 1.1)

    # 최종 판정 (임계값 조정)
    is_deepfake = ensemble_confidence > 50

    details = {
        "model_confidence": model_confidence,
        "algorithm_score": algo_score * 100,
        "suspicious_algorithm_count": suspicious_count,
        "ensemble_confidence": ensemble_confidence,
        "weights": {
            "model": model_weight,
            "algorithms": total_algo_weight,
        }
    }

    return ensemble_confidence, is_deepfake, details


class DeepfakeExplainerService:
    """딥페이크 탐지 및 히트맵 생성 서비스 (GenD-PE)"""

    def __init__(self):
        self._model = None
        self._device = None
        self._initialized = False
        self._init_failed = False

    def _ensure_initialized(self):
        """모델 초기화 (지연 로딩)"""
        if self._initialized:
            return
        if self._init_failed:
            raise RuntimeError("GenD-PE initialization previously failed")

        try:
            from transformers import AutoModel

            # GPU/CPU 설정
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self._device}")

            # GenD-PE 모델 로드
            self._model = AutoModel.from_pretrained(
                "yermandy/GenD_PE_L", trust_remote_code=True
            )
            self._model.eval().to(self._device)

            self._initialized = True
            logger.info("DeepfakeExplainerService (GenD-PE) initialized successfully")

        except Exception as e:
            self._init_failed = True
            self._model = None
            logger.error(f"Failed to initialize DeepfakeExplainerService: {e}")
            raise

    def _generate_attention_heatmap(self, image: Image.Image) -> np.ndarray | None:
        """
        Attention Rollout 기반 히트맵 생성
        EVA ViT backbone의 attention weights를 활용
        """
        try:
            # attention weights 수집을 위한 hook 등록
            attention_maps = []

            def hook_fn(module, input, output):
                # ViT의 attention layer에서 attention weights 추출
                if hasattr(output, 'attn_weights') and output.attn_weights is not None:
                    attention_maps.append(output.attn_weights.detach().cpu())

            hooks = []
            # EVA ViT backbone의 attention layer에 hook 등록
            backbone = None
            if hasattr(self._model, 'backbone'):
                backbone = self._model.backbone
            elif hasattr(self._model, 'model'):
                backbone = self._model.model
            elif hasattr(self._model, 'encoder'):
                backbone = self._model.encoder

            if backbone is not None:
                for name, module in backbone.named_modules():
                    if 'attn' in name.lower() and hasattr(module, 'forward'):
                        # attention dropout이나 softmax 뒤에 hook
                        if 'drop' in name.lower() or name.endswith('attn'):
                            hooks.append(module.register_forward_hook(hook_fn))

            # 추론 실행 (hook으로 attention 수집)
            preprocessed = self._model.feature_extractor.preprocess(image)
            tensor = preprocessed.unsqueeze(0).to(self._device)
            with torch.no_grad():
                self._model(tensor)

            # hook 제거
            for h in hooks:
                h.remove()

            if not attention_maps:
                # hook으로 수집 실패 시, 간단한 gradient 기반 히트맵 생성
                return self._generate_gradient_heatmap(image)

            # Attention Rollout
            result = torch.eye(attention_maps[0].shape[-1])
            for attn in attention_maps:
                # head 평균
                attn_mean = attn.mean(dim=1)[0]  # [seq_len, seq_len]
                # residual 추가
                attn_residual = 0.5 * attn_mean + 0.5 * torch.eye(attn_mean.shape[0])
                # 정규화
                attn_residual = attn_residual / attn_residual.sum(dim=-1, keepdim=True)
                # 누적 곱
                result = torch.matmul(attn_residual, result)

            # CLS → patch attention 추출
            cls_attention = result[0, 1:]  # CLS 토큰에서 patch로의 attention

            # grid reshape
            num_patches = cls_attention.shape[0]
            grid_size = int(num_patches ** 0.5)
            if grid_size * grid_size != num_patches:
                return self._generate_gradient_heatmap(image)

            heatmap = cls_attention.reshape(grid_size, grid_size).numpy()

            # 이미지 크기로 업샘플
            heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)

            # 정규화
            heatmap_min = heatmap.min()
            heatmap_max = heatmap.max()
            if heatmap_max > heatmap_min:
                heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
            else:
                heatmap = np.zeros_like(heatmap)

            return heatmap

        except Exception as e:
            logger.warning(f"Attention heatmap generation failed: {e}")
            return self._generate_gradient_heatmap(image)

    def _generate_gradient_heatmap(self, image: Image.Image) -> np.ndarray | None:
        """Gradient 기반 간단한 히트맵 (폴백)"""
        try:
            img_array = np.array(image.resize((224, 224)))
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Laplacian + Sobel 기반 히트맵
            laplacian = np.abs(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F))
            sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
            sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))

            heatmap = (laplacian * 0.5 + sobelx * 0.25 + sobely * 0.25)
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

            # 정규화
            hm_min = heatmap.min()
            hm_max = heatmap.max()
            if hm_max > hm_min:
                heatmap = (heatmap - hm_min) / (hm_max - hm_min)
            else:
                heatmap = np.zeros_like(heatmap)

            return heatmap.astype(np.float32)
        except Exception as e:
            logger.warning(f"Gradient heatmap generation failed: {e}")
            return None

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

    def _extract_markers_with_landmarks(
        self,
        image_data: bytes,
        heatmap: np.ndarray,
        is_deepfake: bool,
        algorithm_checks: list[AlgorithmCheck],
        top_k: int = 3
    ) -> list[DetectionMarker]:
        """
        의심 영역에만 마커 표시
        딥페이크가 아니거나 강도가 낮으면 마커 없음
        """
        from ..face_landmark import get_face_landmark_service

        markers = []

        # 딥페이크가 아니면 마커 없음
        if not is_deepfake:
            return markers

        # 히트맵 블러링 (노이즈 제거)
        heatmap_blur = cv2.GaussianBlur(heatmap, (11, 11), 0)
        h, w = heatmap_blur.shape

        # 알고리즘 한글 설명 매핑
        algo_korean = {
            "frequency_analysis": "GAN 주파수 패턴",
            "skin_texture": "피부 텍스처 이상",
            "color_consistency": "색상 불일치",
            "edge_artifacts": "경계 아티팩트",
            "noise_pattern": "노이즈 불일치",
            "compression_artifacts": "압축 패턴 이상",
        }

        # 실패한 알고리즘 목록
        failed_algos = [
            algo_korean.get(c.name, c.name)
            for c in algorithm_checks if not c.passed
        ]
        algo_flags = [c.description for c in algorithm_checks if not c.passed]

        # 의심 영역만 찾기 (임계값 0.5 이상)
        hotspots = self._find_suspicious_hotspots(heatmap_blur, min_intensity=0.5, min_distance=30)

        # 의심 영역 없으면 마커 없음
        if not hotspots:
            return markers

        # 얼굴 랜드마크로 라벨링
        face_service = get_face_landmark_service()
        landmark_result = face_service.detect_landmarks(image_data)

        landmarks_dict = {}
        if landmark_result.success and landmark_result.landmarks:
            for lm in landmark_result.landmarks:
                landmarks_dict[lm.label] = (lm.x, lm.y)

        # 상위 의심 영역만 마커 표시 (최대 top_k개)
        for i, (x_pct, y_pct, intensity) in enumerate(hotspots[:top_k]):
            label = self._find_nearest_landmark_label(x_pct, y_pct, landmarks_dict)

            # 설명 생성
            if failed_algos and intensity > 0.6:
                issues_text = ", ".join(failed_algos[:2])
                description = f"AI 조작 감지 - {issues_text} (강도: {intensity:.0%})"
            elif intensity > 0.7:
                description = f"높은 이상 패턴 감지 (강도: {intensity:.0%})"
            else:
                description = f"AI 조작 의심 영역 (강도: {intensity:.0%})"

            markers.append(DetectionMarker(
                id=i + 1,
                x=round(x_pct, 1),
                y=round(y_pct, 1),
                intensity=float(intensity),
                label=label,
                description=description,
                algorithm_flags=algo_flags[:2]
            ))

        return markers

    def _find_suspicious_hotspots(
        self,
        heatmap: np.ndarray,
        min_intensity: float = 0.5,
        min_distance: int = 30
    ) -> list[tuple[float, float, float]]:
        """
        의심 영역만 찾기 (임계값 이상)
        반환: [(x_percent, y_percent, intensity), ...]
        """
        h, w = heatmap.shape
        hotspots = []

        # 로컬 최대값 필터
        local_max = maximum_filter(heatmap, size=15)
        # 임계값 이상이고 로컬 최대값인 위치만
        peaks = (heatmap == local_max) & (heatmap >= min_intensity)

        peak_coords = np.argwhere(peaks)
        if len(peak_coords) == 0:
            return []

        # 강도순 정렬
        peak_intensities = [(y, x, heatmap[y, x]) for y, x in peak_coords]
        peak_intensities.sort(key=lambda p: p[2], reverse=True)

        used_positions = []

        for y, x, intensity in peak_intensities:
            # 거리 체크
            too_close = False
            for ux, uy in used_positions:
                if np.sqrt((x - ux)**2 + (y - uy)**2) < min_distance:
                    too_close = True
                    break

            if too_close:
                continue

            used_positions.append((x, y))

            x_pct = (x / w) * 100
            y_pct = (y / h) * 100
            hotspots.append((x_pct, y_pct, intensity))

        return hotspots

    def _find_nearest_landmark_label(
        self,
        x_pct: float,
        y_pct: float,
        landmarks_dict: dict
    ) -> str:
        """좌표에서 가장 가까운 랜드마크 라벨 찾기"""
        if not landmarks_dict:
            return self._get_region_label(x_pct, y_pct)

        min_dist = float('inf')
        nearest_label = None

        for label, (lx, ly) in landmarks_dict.items():
            dist = np.sqrt((x_pct - lx)**2 + (y_pct - ly)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_label = label

        # 너무 멀면 좌표 기반 라벨 사용
        if min_dist > 15:
            return self._get_region_label(x_pct, y_pct)

        return nearest_label or self._get_region_label(x_pct, y_pct)

    def _extract_markers_from_heatmap_fallback(
        self,
        heatmap_blur: np.ndarray,
        is_deepfake: bool,
        algorithm_checks: list[AlgorithmCheck],
        top_k: int = 3
    ) -> list[DetectionMarker]:
        """의심 영역에만 마커 표시 (폴백)"""
        markers = []

        # 딥페이크가 아니면 마커 없음
        if not is_deepfake:
            return markers

        # 알고리즘 한글 설명 매핑
        algo_korean = {
            "frequency_analysis": "GAN 주파수 패턴",
            "skin_texture": "피부 텍스처 이상",
            "color_consistency": "색상 불일치",
            "edge_artifacts": "경계 아티팩트",
            "noise_pattern": "노이즈 불일치",
            "compression_artifacts": "압축 패턴 이상",
        }

        failed_algos = [
            algo_korean.get(c.name, c.name)
            for c in algorithm_checks if not c.passed
        ]
        algo_flags = [c.description for c in algorithm_checks if not c.passed][:2]

        # 의심 영역만 찾기
        hotspots = self._find_suspicious_hotspots(heatmap_blur, min_intensity=0.5, min_distance=30)

        if not hotspots:
            return markers

        for i, (x_pct, y_pct, intensity) in enumerate(hotspots[:top_k]):
            label = self._get_region_label(x_pct, y_pct)

            if failed_algos and intensity > 0.6:
                issues_text = ", ".join(failed_algos[:2])
                description = f"AI 조작 감지 - {issues_text} (강도: {intensity:.0%})"
            elif intensity > 0.7:
                description = f"높은 이상 패턴 감지 (강도: {intensity:.0%})"
            else:
                description = f"AI 조작 의심 영역 (강도: {intensity:.0%})"

            markers.append(DetectionMarker(
                id=i + 1,
                x=round(x_pct, 1),
                y=round(y_pct, 1),
                intensity=float(intensity),
                label=label,
                description=description,
                algorithm_flags=algo_flags
            ))

        return markers

    def _get_region_label(self, x: float, y: float) -> str:
        """좌표에 따른 영역 라벨 (폴백용)"""
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
        """
        이미지 분석 및 히트맵 생성
        GenD-PE 모델 + 다중 알고리즘 검사 + 앙상블 스코어링
        """
        self._ensure_initialized()

        try:
            # 이미지 로드
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image_array = np.array(image)

            # 1. GenD-PE 모델 예측
            preprocessed = self._model.feature_extractor.preprocess(image)
            tensor = preprocessed.unsqueeze(0).to(self._device)
            with torch.no_grad():
                logits = self._model(tensor)
                probs = logits.softmax(dim=-1)
                fake_prob = probs[0, 1].item()  # class 1 = fake

            model_confidence = fake_prob * 100

            # 2. 다중 알고리즘 검사 실행
            logger.info("Running multi-algorithm deepfake checks...")
            algorithm_checks = run_all_algorithm_checks(image_array)

            # 3. 앙상블 스코어 계산
            ensemble_confidence, is_deepfake, ensemble_details = calculate_ensemble_score(
                model_confidence, algorithm_checks
            )

            logger.info(
                f"Model: {model_confidence:.1f}%, "
                f"Algorithm: {ensemble_details['algorithm_score']:.1f}%, "
                f"Ensemble: {ensemble_confidence:.1f}%, "
                f"Suspicious checks: {ensemble_details['suspicious_algorithm_count']}"
            )

            # 4. 히트맵 생성 (Attention Rollout)
            try:
                heatmap = self._generate_attention_heatmap(image)
            except Exception as e:
                logger.warning(f"Heatmap generation failed, using gradient fallback: {e}")
                heatmap = self._generate_gradient_heatmap(image)

            heatmap_base64 = None
            if heatmap is not None:
                # 5. 얼굴 랜드마크 기반 동적 마커 추출
                markers = self._extract_markers_with_landmarks(
                    image_data, heatmap, is_deepfake, algorithm_checks, top_k=3
                )

                # 6. 히트맵 오버레이 이미지 생성
                image_resized = cv2.resize(image_array, (224, 224))
                overlay = self._overlay_heatmap(image_resized, heatmap)

                # Base64 인코딩
                import base64
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
            else:
                markers = []

            return ExplainerResult(
                is_deepfake=is_deepfake,
                confidence=ensemble_confidence,
                markers=markers,
                heatmap_base64=heatmap_base64,
                raw_heatmap=heatmap,
                algorithm_checks=algorithm_checks,
                ensemble_details=ensemble_details
            )

        except Exception as e:
            logger.error(f"Deepfake analysis failed: {e}")
            raise

    def is_available(self) -> bool:
        """서비스 사용 가능 여부"""
        try:
            import transformers  # noqa: F401
            import timm  # noqa: F401
            return True
        except ImportError:
            return False


# 싱글톤
_service: DeepfakeExplainerService | None = None


def get_deepfake_explainer_service() -> DeepfakeExplainerService:
    """DeepfakeExplainerService 싱글톤"""
    global _service
    if _service is None:
        _service = DeepfakeExplainerService()
    return _service
