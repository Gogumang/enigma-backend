"""
Explainable Deepfake Detector
GenD-PE 기반 딥페이크 탐지 및 히트맵 생성
"""
from .service import DeepfakeExplainerService, get_deepfake_explainer_service

__all__ = ["DeepfakeExplainerService", "get_deepfake_explainer_service"]
