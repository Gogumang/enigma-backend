from unittest.mock import patch, AsyncMock, MagicMock
import pytest
import httpx

from src.infrastructure.external.cybercop_client import CybercopClient


class TestCybercopClient:
    @pytest.fixture
    def client(self):
        return CybercopClient()

    async def test_not_configured(self, client: CybercopClient):
        with patch("src.infrastructure.external.cybercop_client.get_settings") as mock:
            mock.return_value = MagicMock(cybercop_url="")
            result = await client.search("PHONE", "01012345678")
        assert result.found is False
        assert result.error == "사이버캅 API 미설정"

    async def test_found_fraud(self, client: CybercopClient):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "7"

        with patch("src.infrastructure.external.cybercop_client.get_settings") as mock_settings, \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            mock_settings.return_value = MagicMock(cybercop_url="https://cybercop.cyber.go.kr")
            result = await client.search("PHONE", "01012345678")

        assert result.found is True
        assert result.count == 7

    async def test_no_fraud(self, client: CybercopClient):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "0"

        with patch("src.infrastructure.external.cybercop_client.get_settings") as mock_settings, \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            mock_settings.return_value = MagicMock(cybercop_url="https://cybercop.cyber.go.kr")
            result = await client.search("ACCOUNT", "1234567890123")

        assert result.found is False
        assert result.count == 0

    async def test_non_numeric_response(self, client: CybercopClient):
        """응답이 숫자가 아닌 경우"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "error: invalid request"

        with patch("src.infrastructure.external.cybercop_client.get_settings") as mock_settings, \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            mock_settings.return_value = MagicMock(cybercop_url="https://cybercop.cyber.go.kr")
            result = await client.search("PHONE", "01012345678")

        assert result.found is False
        assert result.count == 0

    async def test_http_error(self, client: CybercopClient):
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("src.infrastructure.external.cybercop_client.get_settings") as mock_settings, \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            mock_settings.return_value = MagicMock(cybercop_url="https://cybercop.cyber.go.kr")
            result = await client.search("PHONE", "01012345678")

        assert result.found is False
        assert "503" in result.error

    async def test_network_exception(self, client: CybercopClient):
        with patch("src.infrastructure.external.cybercop_client.get_settings") as mock_settings, \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")):
            mock_settings.return_value = MagicMock(cybercop_url="https://cybercop.cyber.go.kr")
            result = await client.search("PHONE", "01012345678")

        assert result.found is False
        assert result.error is not None

    async def test_whitespace_in_response(self, client: CybercopClient):
        """응답에 공백이 포함된 경우"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "  3  \n"

        with patch("src.infrastructure.external.cybercop_client.get_settings") as mock_settings, \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            mock_settings.return_value = MagicMock(cybercop_url="https://cybercop.cyber.go.kr")
            result = await client.search("PHONE", "01012345678")

        assert result.found is True
        assert result.count == 3
