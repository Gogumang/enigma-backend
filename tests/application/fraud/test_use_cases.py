import pytest

from src.application.fraud.ports import FraudExternalPort
from src.application.fraud.use_cases import CheckFraudUseCase
from src.domain.fraud import FraudSource, FraudStatus, FraudType


# --- Fake Adapters ---

class FakeAlwaysSafe(FraudExternalPort):
    async def search(self, fraud_type: str, value: str) -> FraudSource:
        return FraudSource(source="fake-safe", found=False, count=0)


class FakeAlwaysDanger(FraudExternalPort):
    def __init__(self, count: int = 3):
        self._count = count

    async def search(self, fraud_type: str, value: str) -> FraudSource:
        return FraudSource(source="fake-danger", found=True, count=self._count)


class FakeErrorAdapter(FraudExternalPort):
    async def search(self, fraud_type: str, value: str) -> FraudSource:
        return FraudSource(source="fake-error", found=False, count=0, error="timeout")


# --- Tests ---

class TestCheckFraudUseCase:
    @pytest.fixture
    def safe_use_case(self):
        return CheckFraudUseCase(adapters=[FakeAlwaysSafe()])

    @pytest.fixture
    def danger_use_case(self):
        return CheckFraudUseCase(adapters=[FakeAlwaysDanger(count=5)])

    @pytest.fixture
    def mixed_use_case(self):
        return CheckFraudUseCase(adapters=[FakeAlwaysSafe(), FakeAlwaysDanger(count=2)])

    async def test_phone_safe(self, safe_use_case: CheckFraudUseCase):
        result = await safe_use_case.execute("PHONE", "010-1234-5678")
        assert result.status == FraudStatus.SAFE
        assert result.fraud_type == FraudType.PHONE
        assert result.value == "01012345678"
        assert result.total_records == 0
        assert "신고된 사기 이력이 없습니다" in result.message

    async def test_phone_danger(self, danger_use_case: CheckFraudUseCase):
        result = await danger_use_case.execute("PHONE", "01099998888")
        assert result.status == FraudStatus.DANGER
        assert result.total_records == 5
        assert "사기 신고 이력이 발견" in result.message

    async def test_account_safe(self, safe_use_case: CheckFraudUseCase):
        result = await safe_use_case.execute("ACCOUNT", "1234567890123", "KB")
        assert result.status == FraudStatus.SAFE
        assert result.bank == "국민은행"

    async def test_account_danger(self, danger_use_case: CheckFraudUseCase):
        result = await danger_use_case.execute("ACCOUNT", "9876543210123")
        assert result.status == FraudStatus.DANGER

    async def test_mixed_adapters_one_danger_sets_danger(self, mixed_use_case: CheckFraudUseCase):
        result = await mixed_use_case.execute("PHONE", "01012345678")
        assert result.status == FraudStatus.DANGER
        assert result.total_records == 2
        assert len(result.sources) == 2

    async def test_pattern_analysis_phone(self, safe_use_case: CheckFraudUseCase):
        result = await safe_use_case.execute("PHONE", "07012345678")
        assert result.pattern_analysis is not None
        assert result.pattern_analysis.phone_type == "인터넷전화 (VoIP)"
        assert any("스팸" in w for w in result.recommendations)

    async def test_pattern_analysis_account(self, safe_use_case: CheckFraudUseCase):
        result = await safe_use_case.execute("ACCOUNT", "1234567890123", "SHINHAN")
        assert result.pattern_analysis is not None
        assert result.bank == "신한은행"

    async def test_invalid_type_raises(self, safe_use_case: CheckFraudUseCase):
        with pytest.raises(ValueError):
            await safe_use_case.execute("INVALID", "12345")

    async def test_normalizes_input(self, safe_use_case: CheckFraudUseCase):
        result = await safe_use_case.execute("PHONE", " 010-1234-5678 ")
        assert result.value == "01012345678"

    async def test_to_dict_structure(self, safe_use_case: CheckFraudUseCase):
        result = await safe_use_case.execute("PHONE", "01012345678")
        d = result.to_dict()
        assert "status" in d
        assert "type" in d
        assert "sources" in d
        assert "message" in d
        assert "recommendations" in d
        assert "additionalLinks" in d

    async def test_error_adapter_stays_safe(self):
        uc = CheckFraudUseCase(adapters=[FakeErrorAdapter()])
        result = await uc.execute("PHONE", "01012345678")
        assert result.status == FraudStatus.SAFE
        assert result.sources[0].error == "timeout"

    async def test_no_adapters(self):
        uc = CheckFraudUseCase(adapters=[])
        result = await uc.execute("PHONE", "01012345678")
        assert result.status == FraudStatus.SAFE
        assert len(result.sources) == 0

    async def test_email_type(self, safe_use_case: CheckFraudUseCase):
        result = await safe_use_case.execute("EMAIL", "test@example.com")
        assert result.fraud_type == FraudType.EMAIL
