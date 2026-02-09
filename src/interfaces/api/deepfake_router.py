from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel

from src.application.deepfake import (
    AnalyzeImageUseCase,
    AnalyzeVideoUseCase,
    DeepfakeAnalysisResult,
)
from src.interfaces.api.dependencies import (
    get_analyze_image_use_case,
    get_analyze_video_use_case,
)

router = APIRouter(prefix="/deepfake", tags=["deepfake"])


class AnalysisResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None


def _result_to_dict(result: DeepfakeAnalysisResult) -> dict:
    # 히트맵 추출 (details에서)
    heatmap_base64 = None
    details_copy = dict(result.details) if result.details else {}
    if "heatmap_base64" in details_copy:
        heatmap_base64 = details_copy.pop("heatmap_base64")

    return {
        "isDeepfake": result.is_deepfake,
        "confidence": result.confidence,
        "riskLevel": result.risk_level,
        "mediaType": result.media_type,
        "message": result.message,
        "details": details_copy,
        "analysisReasons": result.analysis_reasons,
        "markers": result.markers,
        "technicalIndicators": result.technical_indicators,
        "overallAssessment": result.overall_assessment,
        # 히트맵 이미지 (base64)
        "heatmapImage": heatmap_base64,
        # 이미지 품질 정보
        "imageQuality": result.image_quality,
        "blurScore": result.blur_score,
        "qualityWarning": result.quality_warning,
        "isReliable": result.is_reliable,
        # 알고리즘 검사 결과 (신규)
        "algorithmChecks": result.algorithm_checks,
        "ensembleDetails": result.ensemble_details,
        # 저화질 보정 전 원래 confidence
        "originalConfidence": result.details.get("original_confidence"),
    }


@router.post("/analyze/image", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    use_case: AnalyzeImageUseCase = Depends(get_analyze_image_use_case)
):
    """이미지 딥페이크 분석"""
    try:
        image_data = await file.read()
        result = await use_case.execute(image_data)

        return AnalysisResponse(
            success=True,
            data=_result_to_dict(result)
        )

    except Exception as e:
        return AnalysisResponse(success=False, error=str(e))


@router.post("/analyze/video", response_model=AnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    use_case: AnalyzeVideoUseCase = Depends(get_analyze_video_use_case)
):
    """비디오 딥페이크 분석"""
    try:
        video_data = await file.read()
        result = await use_case.execute(video_data)

        return AnalysisResponse(
            success=True,
            data=_result_to_dict(result)
        )

    except Exception as e:
        return AnalysisResponse(success=False, error=str(e))
