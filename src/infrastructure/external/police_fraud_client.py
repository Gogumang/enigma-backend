import logging

import httpx

from src.application.fraud.ports import FraudExternalPort
from src.domain.fraud import FraudSource
from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class PoliceFraudClient(FraudExternalPort):
    """경찰청 사이버범죄 사기 이력 조회 어댑터"""

    # 경찰청 API ftype 매핑 (PHONE→P, ACCOUNT→A, EMAIL→E)
    FTYPE_MAP: dict[str, str] = {
        "PHONE": "P",
        "ACCOUNT": "A",
        "EMAIL": "E",
    }

    async def search(self, fraud_type: str, value: str) -> FraudSource:
        source = FraudSource(source="경찰청 사이버범죄 사기 조회")
        settings = get_settings()

        if not settings.police_fraud_url or not settings.police_fraud_key:
            logger.warning("Police fraud API not configured")
            source.error = "경찰청 API 미설정"
            return source

        ftype = self.FTYPE_MAP.get(fraud_type.upper(), fraud_type)

        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.post(
                    f"{settings.police_fraud_url}/user/cyber/fraud.do",
                    data={
                        "key": settings.police_fraud_key,
                        "no": value,
                        "ftype": ftype,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                if resp.status_code == 200:
                    data = resp.json()
                    # 응답 구조: {"result": true, "value": [{"result": "OK", "count": "3"}]}
                    count = 0
                    values = data.get("value", [])
                    if values and isinstance(values, list):
                        first = values[0]
                        if isinstance(first, dict):
                            raw_count = first.get("count", 0)
                            if isinstance(raw_count, str):
                                count = int(raw_count) if raw_count.isdigit() else 0
                            else:
                                count = int(raw_count)
                    source.count = count
                    if count > 0:
                        source.found = True
                else:
                    logger.warning(f"Police fraud API status {resp.status_code}")
                    source.error = f"API 응답 오류 ({resp.status_code})"

        except Exception as e:
            logger.warning(f"Police fraud API failed: {e}")
            source.error = str(e)

        return source
