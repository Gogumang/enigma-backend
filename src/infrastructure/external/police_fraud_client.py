import logging

import httpx

from src.application.fraud.ports import FraudExternalPort
from src.domain.fraud import FraudSource
from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class PoliceFraudClient(FraudExternalPort):
    """경찰청 사이버범죄 사기 이력 조회 어댑터"""

    async def search(self, fraud_type: str, value: str) -> FraudSource:
        source = FraudSource(source="경찰청 사이버범죄 사기 조회")
        settings = get_settings()

        if not settings.police_fraud_url or not settings.police_fraud_key:
            logger.warning("Police fraud API not configured")
            source.error = "경찰청 API 미설정"
            return source

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{settings.police_fraud_url}/user/cyber/fraud.do",
                    data={
                        "key": settings.police_fraud_key,
                        "no": value,
                        "ftype": fraud_type,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                if resp.status_code == 200:
                    data = resp.json()
                    count = data.get("count", 0)
                    if isinstance(count, str):
                        count = int(count) if count.isdigit() else 0
                    source.count = count
                    if count > 0:
                        source.found = True
                        source.records = data.get("list", [])
                else:
                    logger.warning(f"Police fraud API status {resp.status_code}")
                    source.error = f"API 응답 오류 ({resp.status_code})"

        except Exception as e:
            logger.warning(f"Police fraud API failed: {e}")
            source.error = str(e)

        return source
