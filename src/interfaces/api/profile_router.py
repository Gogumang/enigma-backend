
from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from src.application.profile import ProfileSearchUseCase, ReportScammerUseCase
from src.interfaces.api.dependencies import (
    get_profile_search_use_case,
    get_report_scammer_use_case,
)

router = APIRouter(prefix="/profile", tags=["profile"])


class SearchResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None


class ReportRequest(BaseModel):
    name: str
    source: str | None = None


@router.post("/search", response_model=SearchResponse)
async def search_profile(
    image: UploadFile | None = File(None),
    query: str | None = Form(None),
    use_case: ProfileSearchUseCase = Depends(get_profile_search_use_case)
):
    """프로필 검색 (이미지 또는 텍스트)"""
    has_image = image is not None
    has_query = query is not None and query.strip() != ""

    if not has_image and not has_query:
        return SearchResponse(
            success=False,
            error="이미지 또는 검색어를 입력해주세요"
        )

    try:
        if has_image:
            image_data = await image.read()
            result = await use_case.search_by_image(image_data, query)
        else:
            result = await use_case.search_by_query(query)

        return SearchResponse(
            success=True,
            data={
                "totalFound": result.total_found,
                "scammerMatches": [
                    {
                        "id": m.scammer_id,
                        "name": m.name,
                        "confidence": m.confidence,
                        "reportCount": m.report_count
                    }
                    for m in result.scammer_matches
                ],
                "reverseSearchLinks": [
                    {
                        "platform": link.platform,
                        "name": link.name,
                        "url": link.url,
                        "icon": link.icon
                    }
                    for link in result.reverse_search_links
                ],
                "webImageResults": [
                    {
                        "title": r.title,
                        "sourceUrl": r.source_url,
                        "imageUrl": r.image_url,
                        "thumbnailUrl": r.thumbnail_url,
                        "platform": r.platform,
                        "matchScore": r.match_score
                    }
                    for r in result.web_image_results
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
                            "matchScore": p.match_score
                        }
                        for p in profiles
                    ]
                    for platform, profiles in result.profile_matches.items()
                }
            }
        )

    except Exception as e:
        return SearchResponse(success=False, error=str(e))


@router.post("/report", response_model=SearchResponse)
async def report_scammer(
    image: UploadFile = File(...),
    name: str = Form(...),
    source: str | None = Form(None),
    use_case: ReportScammerUseCase = Depends(get_report_scammer_use_case)
):
    """스캐머 신고"""
    if not name.strip():
        return SearchResponse(
            success=False,
            error="스캐머 이름이 필요합니다"
        )

    try:
        image_data = await image.read()
        result = await use_case.report(image_data, name, source)

        return SearchResponse(
            success=result.success,
            data={
                "message": result.message,
                "id": result.scammer_id
            } if result.success else None,
            error=None if result.success else result.message
        )

    except Exception as e:
        return SearchResponse(success=False, error=str(e))


@router.get("/status")
async def get_status(
    use_case: ProfileSearchUseCase = Depends(get_profile_search_use_case)
):
    """스캐머 DB 상태 확인"""
    count = await use_case.scammer_repository.count()
    ready = use_case.face_recognition.is_ready()

    return {
        "success": True,
        "data": {
            "ready": ready,
            "count": count
        }
    }
