from unittest.mock import patch, AsyncMock
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

from src.interfaces.api.fraud_router import router
from src.domain.fraud import FraudCheckResult, FraudSource, FraudStatus, FraudType
from src.domain.fraud.value_objects import PhonePattern


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app: FastAPI):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestFraudCheckEndpoint:
    async def test_phone_safe(self, client: AsyncClient):
        fake_result = FraudCheckResult(
            status=FraudStatus.SAFE,
            fraud_type=FraudType.PHONE,
            value="01012345678",
            display_value="01012345678",
            message="현재까지 신고된 사기 이력이 없습니다.",
            pattern_analysis=PhonePattern(is_valid=True, phone_type="휴대전화"),
        )
        fake_result.additional_links = []

        with patch("src.interfaces.api.fraud_router._use_case") as mock_uc:
            mock_uc.execute = AsyncMock(return_value=fake_result)
            resp = await client.post("/fraud/check", json={
                "type": "PHONE",
                "value": "010-1234-5678",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["status"] == "safe"

    async def test_account_danger(self, client: AsyncClient):
        fake_result = FraudCheckResult(
            status=FraudStatus.DANGER,
            fraud_type=FraudType.ACCOUNT,
            value="1234567890123",
            display_value="1234567890123",
            total_records=3,
            message="⚠️ 사기 신고 이력이 발견되었습니다! (3건)",
        )
        fake_result.additional_links = []

        with patch("src.interfaces.api.fraud_router._use_case") as mock_uc:
            mock_uc.execute = AsyncMock(return_value=fake_result)
            resp = await client.post("/fraud/check", json={
                "type": "ACCOUNT",
                "value": "1234567890123",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["status"] == "danger"
        assert data["data"]["totalRecords"] == 3

    async def test_invalid_type(self, client: AsyncClient):
        with patch("src.interfaces.api.fraud_router._use_case") as mock_uc:
            mock_uc.execute = AsyncMock(side_effect=ValueError("지원하는 조회 유형: PHONE"))
            resp = await client.post("/fraud/check", json={
                "type": "INVALID",
                "value": "12345",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "지원하는 조회 유형" in data["error"]

    async def test_display_value_preserved(self, client: AsyncClient):
        fake_result = FraudCheckResult(
            status=FraudStatus.SAFE,
            fraud_type=FraudType.PHONE,
            value="01012345678",
            display_value="01012345678",
            message="safe",
        )
        fake_result.additional_links = []

        with patch("src.interfaces.api.fraud_router._use_case") as mock_uc:
            mock_uc.execute = AsyncMock(return_value=fake_result)
            resp = await client.post("/fraud/check", json={
                "type": "PHONE",
                "value": "010-1234-5678",
            })

        data = resp.json()
        # displayValue는 원본 형식 유지
        assert data["data"]["displayValue"] == "010-1234-5678"
