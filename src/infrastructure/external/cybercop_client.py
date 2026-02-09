import logging

import httpx

from src.application.fraud.ports import FraudExternalPort
from src.domain.fraud import FraudSource
from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class CybercopClient(FraudExternalPort):
    """사이버캅 사기 이력 조회 어댑터"""

    async def search(self, fraud_type: str, value: str) -> FraudSource:
        source = FraudSource(source="사이버캅")
        settings = get_settings()

        if not settings.cybercop_url:
            logger.warning("Cybercop API not configured")
            source.error = "사이버캅 API 미설정"
            return source

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{settings.cybercop_url}/countFraud.do",
                    params={
                        "fieldType": fraud_type,
                        "keyword": value,
                        "accessType": "3",
                    },
                )

                if resp.status_code == 200:
                    body = resp.text.strip()
                    try:
                        count = int(body)
                    except ValueError:
                        count = 0
                    source.count = count
                    if count > 0:
                        source.found = True
                else:
                    logger.warning(f"Cybercop API status {resp.status_code}")
                    source.error = f"API 응답 오류 ({resp.status_code})"

        except Exception as e:
            logger.warning(f"Cybercop API failed: {e}")
            source.error = str(e)

        return source
