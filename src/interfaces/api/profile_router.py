"""
프로필 검색 API - 강화된 버전
역이미지 검색, 얼굴 검색, 소셜 미디어 검색 통합
"""
import base64
import hashlib
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from src.application.profile import ProfileSearchUseCase, ReportScammerUseCase
from src.infrastructure.persistence import (
    ScammerNetworkRepository,
    ScammerReport,
    SnsProfile,
)
from src.interfaces.api.dependencies import (
    get_profile_search_use_case,
    get_report_scammer_use_case,
    get_scammer_network_repository,
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


@router.post("/search/phone", response_model=SearchResponse)
async def search_by_phone(
    phone: str = Form(...),
    use_case: ProfileSearchUseCase = Depends(get_profile_search_use_case)
):
    """
    전화번호로 검색

    Truecaller, Whoscall 등에서 전화번호 소유자 확인 링크 제공
    """
    if not phone.strip():
        return SearchResponse(
            success=False,
            error="전화번호를 입력해주세요"
        )

    try:
        links = await use_case.search_by_phone(phone)

        return SearchResponse(
            success=True,
            data={
                "phone": phone,
                "searchLinks": links
            }
        )

    except Exception as e:
        return SearchResponse(success=False, error=str(e))


@router.post("/search/email", response_model=SearchResponse)
async def search_by_email(
    email: str = Form(...),
    use_case: ProfileSearchUseCase = Depends(get_profile_search_use_case)
):
    """
    이메일로 검색

    Have I Been Pwned, Epieos 등에서 이메일 정보 확인 링크 제공
    """
    if not email.strip():
        return SearchResponse(
            success=False,
            error="이메일을 입력해주세요"
        )

    try:
        links = await use_case.search_by_email(email)

        return SearchResponse(
            success=True,
            data={
                "email": email,
                "searchLinks": links
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


@router.get("/search-services")
async def get_search_services():
    """사용 가능한 검색 서비스 목록"""
    return {
        "success": True,
        "data": {
            "description": "프로필 검색에 사용 가능한 외부 서비스들",
            "categories": {
                "reverse_image": {
                    "name": "역이미지 검색",
                    "description": "이미지로 원본 출처와 유사 이미지 찾기",
                    "services": ["Google Lens", "Yandex Images", "Bing Visual", "TinEye"]
                },
                "face_search": {
                    "name": "얼굴 검색",
                    "description": "얼굴 인식으로 SNS 프로필 찾기",
                    "services": ["PimEyes", "FaceCheck.ID", "Search4faces", "Social Catfish"]
                },
                "scam_check": {
                    "name": "스캠 확인",
                    "description": "알려진 스캐머 데이터베이스 조회",
                    "services": ["ScamDigger", "ScamWarners", "Romance Scams Now"]
                },
                "social_media": {
                    "name": "소셜 미디어",
                    "description": "이름으로 SNS 프로필 검색",
                    "services": ["Instagram", "Facebook", "LinkedIn", "X", "TikTok", "VK"]
                },
                "contact_info": {
                    "name": "연락처 정보",
                    "description": "전화번호, 이메일 조회",
                    "services": ["Truecaller", "Whoscall", "Have I Been Pwned", "Epieos"]
                }
            },
            "tips": [
                "이미지 검색 시 SerpApi 키를 설정하면 Google Lens 결과를 직접 받을 수 있습니다",
                "PimEyes와 FaceCheck.ID는 유료지만 가장 정확한 얼굴 검색을 제공합니다",
                "Yandex Images는 러시아/동유럽 스캐머 추적에 효과적입니다",
                "스캐머 DB에 등록되지 않은 새로운 스캐머는 /profile/scammer/report로 신고해주세요"
            ]
        }
    }


# ==================== 스캐머 관리 API ====================


class SnsProfileInput(BaseModel):
    """SNS 프로필 입력"""
    platform: str
    profile_url: str
    profile_name: str | None = None
    username: str | None = None
    image_url: str | None = None


class ScammerReportInput(BaseModel):
    """스캐머 신고 입력 (검색 결과 기반)"""
    profile_name: str
    platform: str = "unknown"
    description: str | None = None
    sns_profiles: list[SnsProfileInput] = []
    found_image_urls: list[str] = []
    phone_numbers: list[str] = []
    bank_accounts: list[str] = []
    damage_amount: int | None = None


@router.post("/scammer/report", response_model=SearchResponse)
async def report_scammer_with_sns(
    image: UploadFile = File(...),
    report_data: str = Form(...),  # JSON string of ScammerReportInput
    network_repo: ScammerNetworkRepository = Depends(get_scammer_network_repository),
    profile_use_case: ProfileSearchUseCase = Depends(get_profile_search_use_case)
):
    """
    스캐머 신고 (SNS 프로필 정보 포함)

    검색 결과에서 스캐머로 판단된 경우 호출
    - 원본 이미지
    - 발견된 SNS 프로필들
    - 발견된 이미지 URLs
    """
    import json

    try:
        # JSON 파싱
        data = json.loads(report_data)
        input_data = ScammerReportInput(**data)

        # 이미지 읽기
        image_data = await image.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # 얼굴 임베딩 추출
        face_embedding = None
        try:
            embedding = await profile_use_case.face_recognition.extract_embedding(image_data)
            if embedding:
                face_embedding = list(embedding)
        except Exception:
            pass  # 얼굴 추출 실패해도 계속 진행

        # 이미지 해시 생성
        photo_hash = hashlib.sha256(image_data).hexdigest()[:32]

        # 스캐머 ID 생성
        scammer_id = f"scammer-{uuid.uuid4().hex[:12]}"

        # SNS 프로필 변환
        sns_profiles = [
            SnsProfile(
                platform=sp.platform,
                profile_url=sp.profile_url,
                profile_name=sp.profile_name,
                username=sp.username,
                image_url=sp.image_url
            )
            for sp in input_data.sns_profiles
        ]

        # 스캐머 신고 생성
        report = ScammerReport(
            id=scammer_id,
            platform=input_data.platform,
            profile_name=input_data.profile_name,
            description=input_data.description or "",
            reported_at=datetime.now(),
            phone_numbers=input_data.phone_numbers,
            bank_accounts=input_data.bank_accounts,
            profile_photo_hash=photo_hash,
            damage_amount=input_data.damage_amount,
            sns_profiles=sns_profiles,
            found_image_urls=input_data.found_image_urls,
            original_image_base64=image_base64,
            face_embedding=face_embedding
        )

        # Neo4j에 저장
        saved_id = await network_repo.report_scammer(report)

        return SearchResponse(
            success=True,
            data={
                "message": "스캐머가 등록되었습니다",
                "scammerId": saved_id,
                "snsProfileCount": len(sns_profiles),
                "foundImageCount": len(input_data.found_image_urls),
            }
        )

    except json.JSONDecodeError:
        return SearchResponse(success=False, error="잘못된 JSON 형식입니다")
    except Exception as e:
        return SearchResponse(success=False, error=str(e))


@router.get("/scammer/list", response_model=SearchResponse)
async def list_scammers(
    limit: int = 50,
    offset: int = 0,
    network_repo: ScammerNetworkRepository = Depends(get_scammer_network_repository)
):
    """등록된 스캐머 목록 조회"""
    try:
        scammers = await network_repo.list_scammers(limit=limit, offset=offset)

        return SearchResponse(
            success=True,
            data={
                "scammers": scammers,
                "count": len(scammers),
                "limit": limit,
                "offset": offset
            }
        )

    except Exception as e:
        return SearchResponse(success=False, error=str(e))


@router.get("/scammer/{scammer_id}", response_model=SearchResponse)
async def get_scammer_detail(
    scammer_id: str,
    network_repo: ScammerNetworkRepository = Depends(get_scammer_network_repository)
):
    """스캐머 상세 정보 조회"""
    try:
        scammer = await network_repo.get_scammer_detail(scammer_id)

        if not scammer:
            return SearchResponse(success=False, error="스캐머를 찾을 수 없습니다")

        return SearchResponse(
            success=True,
            data=scammer
        )

    except Exception as e:
        return SearchResponse(success=False, error=str(e))


@router.get("/scammer/search/sns", response_model=SearchResponse)
async def search_scammer_by_sns(
    url: str,
    network_repo: ScammerNetworkRepository = Depends(get_scammer_network_repository)
):
    """SNS URL로 스캐머 검색"""
    try:
        scammers = await network_repo.find_by_sns_url(url)

        return SearchResponse(
            success=True,
            data={
                "scammers": scammers,
                "count": len(scammers),
                "searchUrl": url
            }
        )

    except Exception as e:
        return SearchResponse(success=False, error=str(e))


@router.get("/scammer/network/{scammer_id}", response_model=SearchResponse)
async def get_scammer_network(
    scammer_id: str,
    network_repo: ScammerNetworkRepository = Depends(get_scammer_network_repository)
):
    """스캐머 네트워크 분석"""
    try:
        analysis = await network_repo.analyze_network(scammer_id)

        return SearchResponse(
            success=True,
            data={
                "scammerId": scammer_id,
                "relatedScammers": analysis.related_scammers,
                "sharedAccounts": analysis.shared_accounts,
                "sharedPhones": analysis.shared_phones,
                "networkSize": analysis.network_size,
                "riskLevel": analysis.risk_level,
                "warnings": analysis.warnings
            }
        )

    except Exception as e:
        return SearchResponse(success=False, error=str(e))


@router.get("/scammer/stats", response_model=SearchResponse)
async def get_scammer_stats(
    network_repo: ScammerNetworkRepository = Depends(get_scammer_network_repository)
):
    """스캐머 네트워크 통계"""
    try:
        stats = await network_repo.get_network_stats()
        largest = await network_repo.get_largest_networks(limit=5)

        return SearchResponse(
            success=True,
            data={
                "stats": stats,
                "largestNetworks": largest
            }
        )

    except Exception as e:
        return SearchResponse(success=False, error=str(e))
