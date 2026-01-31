from fastapi import APIRouter
from pydantic import BaseModel
import httpx

router = APIRouter(prefix="/fraud", tags=["fraud"])


class FraudCheckRequest(BaseModel):
    type: str  # PHONE, ACCOUNT, EMAIL
    value: str


class FraudCheckResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None


@router.post("/check", response_model=FraudCheckResponse)
async def check_fraud(request: FraudCheckRequest):
    """전화번호, 계좌번호, 이메일의 사기 이력 조회"""
    try:
        check_type = request.type.upper()
        value = request.value.replace("-", "").strip()

        if check_type not in ["PHONE", "ACCOUNT", "EMAIL"]:
            return FraudCheckResponse(
                success=False,
                error="올바른 조회 유형이 아닙니다 (PHONE, ACCOUNT, EMAIL)"
            )

        # 외부 사기 조회 API 호출 (thecheat.co.kr API 등)
        # 현재는 시뮬레이션 응답
        async with httpx.AsyncClient() as client:
            try:
                # 더치트 API 시뮬레이션
                # 실제로는 https://thecheat.co.kr/rb/?mod=_search API 등을 호출
                is_fraud = False
                records = []

                # 테스트용: 특정 패턴이면 사기 이력 있음으로 표시
                if value.endswith("1234") or "scam" in value.lower():
                    is_fraud = True
                    records = [
                        {
                            "type": "로맨스 스캠",
                            "date": "2024-01",
                            "description": "온라인 만남 후 투자 명목 금전 요구"
                        },
                        {
                            "type": "보이스피싱",
                            "date": "2023-11",
                            "description": "기관 사칭 금전 요구"
                        }
                    ]

                return FraudCheckResponse(
                    success=True,
                    data={
                        "status": "danger" if is_fraud else "safe",
                        "type": check_type,
                        "value": value,
                        "records": records,
                        "message": "사기 이력이 발견되었습니다." if is_fraud else "사기 이력이 없습니다."
                    }
                )

            except Exception as e:
                return FraudCheckResponse(
                    success=False,
                    error=f"외부 API 호출 실패: {str(e)}"
                )

    except Exception as e:
        return FraudCheckResponse(success=False, error=str(e))
