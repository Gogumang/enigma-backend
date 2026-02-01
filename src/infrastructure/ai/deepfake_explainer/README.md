# Deepfake Explainer Model Setup

EfficientViT 기반 딥페이크 탐지 모델 설정 가이드

## 모델 다운로드

1. [Google Drive](https://drive.google.com/drive/folders/1-JtWGMyd7YaTa56R6uYpjvwmUyW5q-zN)에서 모델 파일 다운로드

2. 다운로드한 파일을 `models/` 폴더에 저장:
   ```
   src/infrastructure/ai/deepfake_explainer/models/efficientnetB0_checkpoint89_All
   ```

## 폴더 구조

```
deepfake_explainer/
├── models/
│   └── efficientnetB0_checkpoint89_All  # 모델 파일 (다운로드 필요)
├── transforms/
│   └── albu.py
├── evit_model.py
├── explained_architecture.yaml
├── service.py
└── __init__.py
```

## 사용법

모델 파일이 있으면 자동으로 EfficientViT 분석이 활성화됩니다.
모델 파일이 없으면 기존 방식(Sightengine + OpenAI)으로 fallback됩니다.

## 요구사항

- GPU 권장 (CUDA)
- CPU에서도 작동하지만 느릴 수 있음
