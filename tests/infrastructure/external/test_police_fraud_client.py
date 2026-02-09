from unittest.mock import patch, AsyncMock, MagicMock
import pytest
import httpx

from src.infrastructure.external.police_fraud_client import PoliceFraudClient


class TestPoliceFraudClient:
    @pytest.fixture
    def client(self):
        return PoliceFraudClient()

    async def test_not_configured(self, client: PoliceFraudClient):
        """API 미설정 시 에러 메시지 반환"""
        with patch("src.infrastructure.external.police_fraud_client.get_settings") as mock:
            mock.return_value = MagicMock(police_fraud_url="", police_fraud_key="")
            result = await client.search("PHONE", "01012345678")
        assert result.found is False
        assert result.error == "경찰청 API 미설정"

    async def test_found_fraud(self, client: PoliceFraudClient):
        """사기 이력 발견 시"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "count": 3,
            "list": [{"id": 1}, {"id": 2}, {"id": 3}],
        }

        with patch("src.infrastructure.external.police_fraud_client.get_settings") as mock_settings, \
             patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            mock_settings.return_value = MagicMock(
                police_fraud_url="https://www.police.go.kr",
                police_fraud_key="P",
            )
            result = await client.search("PHONE", "01012345678")

        assert result.found is True
        assert result.count == 3
        assert len(result.records) == 3

    async def test_no_fraud(self, client: PoliceFraudClient):
        """사기 이력 없음"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"count": 0, "list": []}

        with patch("src.infrastructure.external.police_fraud_client.get_settings") as mock_settings, \
             patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            mock_settings.return_value = MagicMock(
                police_fraud_url="https://www.police.go.kr",
                police_fraud_key="P",
            )
            result = await client.search("ACCOUNT", "1234567890123")

        assert result.found is False
        assert result.count == 0

    async def test_count_as_string(self, client: PoliceFraudClient):
        """count가 문자열로 오는 경우"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"count": "5", "list": []}

        with patch("src.infrastructure.external.police_fraud_client.get_settings") as mock_settings, \
             patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            mock_settings.return_value = MagicMock(
                police_fraud_url="https://www.police.go.kr",
                police_fraud_key="P",
            )
            result = await client.search("PHONE", "01012345678")

        assert result.found is True
        assert result.count == 5

    async def test_http_error(self, client: PoliceFraudClient):
        """HTTP 에러 응답"""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("src.infrastructure.external.police_fraud_client.get_settings") as mock_settings, \
             patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            mock_settings.return_value = MagicMock(
                police_fraud_url="https://www.police.go.kr",
                police_fraud_key="P",
            )
            result = await client.search("PHONE", "01012345678")

        assert result.found is False
        assert "500" in result.error

    async def test_network_exception(self, client: PoliceFraudClient):
        """네트워크 예외"""
        with patch("src.infrastructure.external.police_fraud_client.get_settings") as mock_settings, \
             patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.ConnectError("connection refused")):
            mock_settings.return_value = MagicMock(
                police_fraud_url="https://www.police.go.kr",
                police_fraud_key="P",
            )
            result = await client.search("PHONE", "01012345678")

        assert result.found is False
        assert result.error is not None
