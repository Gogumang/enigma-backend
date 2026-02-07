"""
사기 신고 API 라우터
- POST /report — 사기 신고 저장
- GET /report/{report_id} — 신고 조회
- POST /report/check — 식별자로 기존 신고 이력 조회
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.infrastructure.persistence import ScamReportRepository
from src.interfaces.api.dependencies import get_scam_report_repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/report", tags=["report"])


# ==================== Request / Response 모델 ====================

class IdentifierItem(BaseModel):
    type: str  # PHONE, ACCOUNT, SNS, URL
    value: str


class ReportRequest(BaseModel):
    overallScore: float = 0
    deepfakeScore: float = 0
    chatScore: float = 0
    fraudScore: float = 0
    urlScore: float = 0
    profileScore: float = 0
    reasons: list[str] = []
    identifiers: list[IdentifierItem] = []
    details: str = ""


class CheckRequest(BaseModel):
    type: str  # PHONE, ACCOUNT, SNS, URL
    value: str


class ReportResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


# ==================== 엔드포인트 ====================

@router.post("", response_model=ReportResponse)
async def create_report(
    request: ReportRequest,
    repo: ScamReportRepository = Depends(get_scam_report_repository),
):
    """사기 신고 저장"""
    try:
        report_id = await repo.save_report(request.model_dump())
        return ReportResponse(
            success=True,
            data={"reportId": report_id, "message": "신고가 저장되었습니다"},
        )
    except Exception as e:
        logger.error(f"Failed to save report: {e}", exc_info=True)
        return ReportResponse(success=False, error=str(e))


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(
    report_id: str,
    repo: ScamReportRepository = Depends(get_scam_report_repository),
):
    """신고 조회"""
    try:
        report = await repo.get_report(report_id)
        if not report:
            return ReportResponse(success=False, error="신고를 찾을 수 없습니다")
        return ReportResponse(success=True, data=report)
    except Exception as e:
        logger.error(f"Failed to get report: {e}", exc_info=True)
        return ReportResponse(success=False, error=str(e))


@router.post("/check", response_model=ReportResponse)
async def check_existing_reports(
    request: CheckRequest,
    repo: ScamReportRepository = Depends(get_scam_report_repository),
):
    """식별자로 기존 신고 이력 조회"""
    try:
        reports = await repo.find_by_identifier(request.type, request.value)
        return ReportResponse(
            success=True,
            data={
                "found": len(reports) > 0,
                "count": len(reports),
                "reports": reports,
            },
        )
    except Exception as e:
        logger.error(f"Failed to check reports: {e}", exc_info=True)
        return ReportResponse(success=False, error=str(e))
