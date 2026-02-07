"""
프로필 검색 API
역이미지 검색, 얼굴 검색, 소셜 미디어 검색 통합
"""

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from src.application.profile import ProfileSearchUseCase
from src.interfaces.api.dependencies import get_profile_search_use_case

router = APIRouter(prefix="/profile", tags=["profile"])


class SearchResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None


@router.post("/search", response_model=SearchResponse)
async def search_profile(
    image: UploadFile | None = File(None),
    query: str | None = Form(None),
    use_case: ProfileSearchUseCase = Depends(get_profile_search_use_case)
):
    """
    프로필 검색 (이미지 또는 텍스트)

    - 이미지: 역이미지 검색 + 얼굴 인식 + 스캐머 DB 비교
    - 텍스트: 소셜 미디어 프로필 검색 링크 생성

    응답:
    - scammerMatches: 스캐머 DB에서 일치하는 항목
    - webImageResults: 역이미지 검색 결과
    - searchLinks: 역이미지/얼굴 검색 서비스 링크 (카테고리별)
    - socialSearchLinks: 소셜 미디어 검색 링크
    """
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
                # 스캐머 DB 매치
                "scammerMatches": [
                    {
                        "id": m.scammer_id,
                        "name": m.name,
                        "confidence": m.confidence,
                        "reportCount": m.report_count
                    }
                    for m in result.scammer_matches
                ],
                # 역이미지 검색 결과 (실제 찾은 이미지들)
                "webImageResults": [
                    {
                        "title": r.title,
                        "sourceUrl": r.source_url,
                        "imageUrl": r.image_url,
                        "thumbnailUrl": r.thumbnail_url,
                        "platform": r.platform,
                        "matchScore": r.match_score,
                        "sourceEngine": r.source_engine
                    }
                    for r in result.web_image_results
                ],
                # 검색 서비스 링크 (카테고리별: reverse_image, face_search, scam_check)
                "searchLinks": [
                    {
                        "name": link.name,
                        "url": link.url,
                        "icon": link.icon,
                        "category": link.category,
                        "description": link.description,
                        "priority": link.priority
                    }
                    for link in result.search_links
                ],
                # 소셜 미디어 검색 링크
                "socialSearchLinks": result.social_search_links,
                # 업로드된 이미지 URL (검색에 사용됨)
                "uploadedImageUrl": result.uploaded_image_url,
                # 하위 호환성: 기존 형식
                "reverseSearchLinks": [
                    {
                        "platform": link.platform,
                        "name": link.name,
                        "url": link.url,
                        "icon": link.icon
                    }
                    for link in result.reverse_search_links
                ],
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
