"""
종합 분석 API — 모든 분석을 한 번에 병렬로 수행
"""
import asyncio
import json
import logging

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from src.interfaces.api.dependencies import (
    get_analyze_chat_use_case,
    get_analyze_image_use_case,
    get_openai_service,
    get_profile_search_use_case,
)
from src.interfaces.api.deepfake_router import _result_to_dict
from src.interfaces.api.fraud_router import perform_fraud_check
from src.interfaces.api.url_router import perform_url_check

router = APIRouter(prefix="/comprehensive", tags=["comprehensive"])
logger = logging.getLogger(__name__)


class ComprehensiveResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None


@router.post("/analyze", response_model=ComprehensiveResponse)
async def comprehensive_analyze(
    image: UploadFile | None = File(None),
    chat_messages: str | None = Form(None),
    chat_screenshot: UploadFile | None = File(None),
    phone: str | None = Form(None),
    account: str | None = Form(None),
    url: str | None = Form(None),
):
    """모든 분석을 한 번에 수행 (입력된 항목만 병렬 실행)"""
    tasks: dict[str, asyncio.Task] = {}
    errors: dict[str, str] = {}

    # 이미지 데이터 선제 읽기 (deepfake + profile 공용)
    image_data: bytes | None = None
    if image is not None:
        image_data = await image.read()

    screenshot_data: bytes | None = None
    if chat_screenshot is not None:
        screenshot_data = await chat_screenshot.read()

    # --- 태스크 정의 ---

    async def run_deepfake() -> dict | None:
        if image_data is None:
            return None
        use_case = get_analyze_image_use_case()
        result = await use_case.execute(image_data)
        return _result_to_dict(result)

    async def run_profile() -> dict | None:
        if image_data is None:
            return None
        # 항상 원본 전체 이미지로 검색 (크롭하면 역이미지 검색 결과가 나빠짐)
        use_case = get_profile_search_use_case()
        result = await use_case.search_by_image(image_data)

        return {
            "totalFound": result.total_found,
            "webImageResults": [
                {
                    "title": r.title,
                    "sourceUrl": r.source_url,
                    "imageUrl": r.image_url,
                    "thumbnailUrl": r.thumbnail_url,
                    "platform": r.platform,
                    "matchScore": r.match_score,
                }
                for r in result.web_image_results
            ],
            "reverseSearchLinks": [
                {
                    "platform": link.platform,
                    "name": link.name,
                    "url": link.url,
                    "icon": link.icon,
                }
                for link in result.reverse_search_links
            ],
            "uploadedImageUrl": result.uploaded_image_url,
            "results": {
                platform: [
                    {
                        "platform": p.platform,
                        "name": p.name,
                        "username": p.username,
                        "profileUrl": p.profile_url,
                        "imageUrl": p.image_url,
                        "matchScore": p.match_score,
                    }
                    for p in profiles
                ]
                for platform, profiles in result.profile_matches.items()
            },
        }

    async def run_chat() -> dict | None:
        # 스크린샷 우선
        if screenshot_data is not None:
            openai_service = get_openai_service()
            result = await openai_service.analyze_chat_screenshot(screenshot_data, None)
            data = {
                "riskScore": result.risk_score,
                "riskCategory": result.risk_category.value,
                "detectedPatterns": result.detected_patterns,
                "warningSigns": result.warning_signs,
                "recommendations": result.recommendations,
                "aiAnalysis": result.ai_analysis,
                "interpretationSteps": result.interpretation_steps,
            }
            if result.parsed_messages:
                data["parsedMessages"] = [
                    {"role": p.role.value, "content": p.content}
                    for p in result.parsed_messages
                ]
            return data

        if chat_messages:
            messages = json.loads(chat_messages)
            if not messages:
                return None
            use_case = get_analyze_chat_use_case()
            result = await use_case.execute(messages=messages)
            data = {
                "riskScore": result.risk_score,
                "riskCategory": result.risk_category,
                "detectedPatterns": result.detected_patterns,
                "warningSigns": result.warning_signs,
                "recommendations": result.recommendations,
                "aiAnalysis": result.ai_analysis,
                "interpretationSteps": result.interpretation_steps,
            }
            if result.rag_context:
                data["ragContext"] = result.rag_context
            if result.parsed_messages:
                data["parsedMessages"] = result.parsed_messages
            return data

        return None

    async def run_fraud() -> dict | None:
        phone_val = (phone or "").strip()
        account_val = (account or "").strip()
        if not phone_val and not account_val:
            return None

        result: dict = {}

        if phone_val:
            result["phone"] = await perform_fraud_check("PHONE", phone_val)

        if account_val:
            result["account"] = await perform_fraud_check("ACCOUNT", account_val)

        return result

    async def run_url() -> dict | None:
        url_val = (url or "").strip()
        if not url_val:
            return None
        return await perform_url_check(url_val)

    # --- 병렬 실행 ---
    task_map = {
        "deepfake": run_deepfake,
        "profile": run_profile,
        "chat": run_chat,
        "fraud": run_fraud,
        "url": run_url,
    }

    async def safe_run(name: str, fn):
        try:
            return await fn()
        except Exception as e:
            logger.warning(f"Comprehensive analysis — {name} failed: {e}", exc_info=True)
            errors[name] = str(e)
            return None

    results = await asyncio.gather(
        *[safe_run(name, fn) for name, fn in task_map.items()]
    )

    data = {}
    for i, name in enumerate(task_map):
        data[name] = results[i]

    if errors:
        data["errors"] = errors

    return ComprehensiveResponse(success=True, data=data)
