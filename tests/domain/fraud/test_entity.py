from src.domain.fraud.entity import FraudCheckResult, FraudSource
from src.domain.fraud.value_objects import FraudStatus, FraudType, PhonePattern


class TestFraudSource:
    def test_default_not_found(self):
        s = FraudSource(source="test")
        assert s.found is False
        assert s.count == 0

    def test_to_dict_minimal(self):
        s = FraudSource(source="경찰청")
        d = s.to_dict()
        assert d == {"source": "경찰청", "found": False, "count": 0}

    def test_to_dict_with_records(self):
        s = FraudSource(source="사이버캅", found=True, count=3, records=["a", "b", "c"])
        d = s.to_dict()
        assert d["found"] is True
        assert d["count"] == 3
        assert len(d["records"]) == 3

    def test_to_dict_with_error(self):
        s = FraudSource(source="경찰청", error="timeout")
        d = s.to_dict()
        assert d["error"] == "timeout"


class TestFraudCheckResult:
    def _make_result(self) -> FraudCheckResult:
        return FraudCheckResult(
            status=FraudStatus.SAFE,
            fraud_type=FraudType.PHONE,
            value="01012345678",
            display_value="010-1234-5678",
        )

    def test_initial_state_is_safe(self):
        r = self._make_result()
        assert r.status == FraudStatus.SAFE
        assert r.total_records == 0

    def test_add_source_safe(self):
        r = self._make_result()
        r.add_source(FraudSource(source="경찰청", found=False, count=0))
        assert r.status == FraudStatus.SAFE
        assert r.total_records == 0
        assert len(r.sources) == 1

    def test_add_source_danger(self):
        r = self._make_result()
        r.add_source(FraudSource(source="경찰청", found=True, count=3))
        assert r.status == FraudStatus.DANGER
        assert r.total_records == 3

    def test_add_multiple_sources_accumulates(self):
        r = self._make_result()
        r.add_source(FraudSource(source="경찰청", found=True, count=2))
        r.add_source(FraudSource(source="사이버캅", found=True, count=5))
        assert r.total_records == 7

    def test_build_message_safe(self):
        r = self._make_result()
        r.build_message()
        assert "신고된 사기 이력이 없습니다" in r.message
        assert len(r.recommendations) > 0
        assert len(r.additional_links) == 2

    def test_build_message_danger(self):
        r = self._make_result()
        r.add_source(FraudSource(source="경찰청", found=True, count=1))
        r.build_message()
        assert "사기 신고 이력이 발견" in r.message
        assert any("경찰(112)" in rec for rec in r.recommendations)

    def test_to_dict(self):
        r = self._make_result()
        r.pattern_analysis = PhonePattern(is_valid=True, phone_type="휴대전화")
        r.build_message()
        d = r.to_dict()
        assert d["status"] == "safe"
        assert d["type"] == "PHONE"
        assert d["value"] == "01012345678"
        assert "patternAnalysis" in d
        assert d["patternAnalysis"]["isValid"] is True

    def test_to_dict_without_pattern(self):
        r = self._make_result()
        r.build_message()
        d = r.to_dict()
        assert "patternAnalysis" not in d
