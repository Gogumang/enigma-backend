"""
스캐머 네트워크 분석 API
- 스캐머 신고
- 네트워크 분석
- 연관 스캐머 검색
"""
from datetime import datetime
from typing import Optional
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.infrastructure.persistence import ScammerNetworkRepository, ScammerReport
from src.interfaces.api.dependencies import get_scammer_network_repository

router = APIRouter(prefix="/network", tags=["scammer-network"])


class ReportRequest(BaseModel):
    """스캐머 신고 요청"""
    platform: str
    profile_name: str
    description: str
    phone_numbers: list[str] = []
    bank_accounts: list[str] = []
    damage_amount: Optional[int] = None
    scam_patterns: list[str] = []


class SearchRequest(BaseModel):
    """검색 요청"""
    query: str
    search_type: str  # account, phone, photo_hash


class ApiResponse(BaseModel):
    """API 응답"""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


@router.post("/report", response_model=ApiResponse)
async def report_scammer(
    request: ReportRequest,
    repo: ScammerNetworkRepository = Depends(get_scammer_network_repository)
):
    """스캐머 신고 및 네트워크에 추가"""
    if not repo.is_connected():
        return ApiResponse(
            success=False,
            error="데이터베이스에 연결되지 않았습니다"
        )

    try:
        report = ScammerReport(
            id=f"scammer-{uuid.uuid4().hex[:8]}",
            platform=request.platform,
            profile_name=request.profile_name,
            description=request.description,
            reported_at=datetime.now(),
            phone_numbers=request.phone_numbers,
            bank_accounts=request.bank_accounts,
            damage_amount=request.damage_amount,
            scam_patterns=request.scam_patterns,
        )

        scammer_id = await repo.report_scammer(report)

        # 신고 후 즉시 네트워크 분석
        network = await repo.analyze_network(scammer_id)

        return ApiResponse(
            success=True,
            data={
                "scammerId": scammer_id,
                "message": "신고가 접수되었습니다",
                "network": {
                    "relatedScammers": network.related_scammers,
                    "networkSize": network.network_size,
                    "riskLevel": network.risk_level,
                    "warnings": network.warnings,
                }
            }
        )

    except Exception as e:
        return ApiResponse(success=False, error=str(e))


@router.get("/analyze/{scammer_id}", response_model=ApiResponse)
async def analyze_network(
    scammer_id: str,
    repo: ScammerNetworkRepository = Depends(get_scammer_network_repository)
):
    """특정 스캐머의 네트워크 분석"""
    if not repo.is_connected():
        return ApiResponse(
            success=False,
            error="데이터베이스에 연결되지 않았습니다"
        )

    try:
        network = await repo.analyze_network(scammer_id)

        return ApiResponse(
            success=True,
            data={
                "scammerId": scammer_id,
                "relatedScammers": network.related_scammers,
                "sharedAccounts": network.shared_accounts,
                "sharedPhones": network.shared_phones,
                "networkSize": network.network_size,
                "riskLevel": network.risk_level,
                "warnings": network.warnings,
            }
        )

    except Exception as e:
        return ApiResponse(success=False, error=str(e))


@router.get("/search/account/{account_number}", response_model=ApiResponse)
async def search_by_account(
    account_number: str,
    repo: ScammerNetworkRepository = Depends(get_scammer_network_repository)
):
    """계좌번호로 스캐머 검색"""
    if not repo.is_connected():
        return ApiResponse(
            success=False,
            error="데이터베이스에 연결되지 않았습니다"
        )

    try:
        scammers = await repo.find_by_account(account_number)

        return ApiResponse(
            success=True,
            data={
                "query": account_number,
                "queryType": "account",
                "results": scammers,
                "count": len(scammers),
                "warning": f"이 계좌를 사용한 {len(scammers)}명의 스캐머가 발견되었습니다" if scammers else None
            }
        )

    except Exception as e:
        return ApiResponse(success=False, error=str(e))


@router.get("/search/phone/{phone_number}", response_model=ApiResponse)
async def search_by_phone(
    phone_number: str,
    repo: ScammerNetworkRepository = Depends(get_scammer_network_repository)
):
    """전화번호로 스캐머 검색"""
    if not repo.is_connected():
        return ApiResponse(
            success=False,
            error="데이터베이스에 연결되지 않았습니다"
        )

    try:
        scammers = await repo.find_by_phone(phone_number)

        return ApiResponse(
            success=True,
            data={
                "query": phone_number,
                "queryType": "phone",
                "results": scammers,
                "count": len(scammers),
                "warning": f"이 번호를 사용한 {len(scammers)}명의 스캐머가 발견되었습니다" if scammers else None
            }
        )

    except Exception as e:
        return ApiResponse(success=False, error=str(e))


@router.get("/stats", response_model=ApiResponse)
async def get_network_stats(
    repo: ScammerNetworkRepository = Depends(get_scammer_network_repository)
):
    """전체 네트워크 통계"""
    if not repo.is_connected():
        return ApiResponse(
            success=False,
            error="데이터베이스에 연결되지 않았습니다"
        )

    try:
        stats = await repo.get_network_stats()
        largest = await repo.get_largest_networks(5)

        return ApiResponse(
            success=True,
            data={
                "totalScammers": stats["scammers"],
                "totalAccounts": stats["accounts"],
                "totalPhones": stats["phones"],
                "totalDamage": stats["total_damage"],
                "largestNetworks": largest,
            }
        )

    except Exception as e:
        return ApiResponse(success=False, error=str(e))


@router.get("/largest", response_model=ApiResponse)
async def get_largest_networks(
    limit: int = 10,
    repo: ScammerNetworkRepository = Depends(get_scammer_network_repository)
):
    """가장 큰 스캐머 네트워크 조회"""
    if not repo.is_connected():
        return ApiResponse(
            success=False,
            error="데이터베이스에 연결되지 않았습니다"
        )

    try:
        networks = await repo.get_largest_networks(limit)

        return ApiResponse(
            success=True,
            data={
                "networks": networks,
                "count": len(networks),
            }
        )

    except Exception as e:
        return ApiResponse(success=False, error=str(e))
