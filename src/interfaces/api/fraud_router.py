import logging

from fastapi import APIRouter
from pydantic import BaseModel

from src.application.fraud import CheckFraudUseCase
from src.infrastructure.external.police_fraud_client import PoliceFraudClient
from src.infrastructure.external.cybercop_client import CybercopClient

router = APIRouter(prefix="/fraud", tags=["fraud"])
logger = logging.getLogger(__name__)

# 어댑터 → 유스케이스 조립
_police = PoliceFraudClient()
_cybercop = CybercopClient()
_use_case = CheckFraudUseCase(adapters=[_police, _cybercop])


def get_fraud_use_case() -> CheckFraudUseCase:
    return _use_case


class FraudCheckRequest(BaseModel):
    type: str  # PHONE, ACCOUNT, EMAIL
    value: str
    bank_code: str | None = None


class FraudCheckResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None


@router.post("/check", response_model=FraudCheckResponse)
async def check_fraud(request: FraudCheckRequest):
    """전화번호, 계좌번호, 이메일의 사기 이력 조회"""
    try:
        result = await _use_case.execute(request.type, request.value, request.bank_code)
        data = result.to_dict()
        data["displayValue"] = request.value  # 원본 형식 유지
        return FraudCheckResponse(success=True, data=data)
    except ValueError as e:
        return FraudCheckResponse(success=False, error=str(e))
    except Exception as e:
        logger.error(f"Fraud check failed: {e}", exc_info=True)
        return FraudCheckResponse(success=False, error=str(e))
