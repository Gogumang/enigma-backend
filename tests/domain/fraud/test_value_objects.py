from src.domain.fraud.value_objects import (
    FraudStatus,
    FraudType,
    PhonePattern,
    AccountPattern,
)


class TestFraudStatus:
    def test_safe_value(self):
        assert FraudStatus.SAFE.value == "safe"

    def test_danger_value(self):
        assert FraudStatus.DANGER.value == "danger"


class TestFraudType:
    def test_phone(self):
        assert FraudType.PHONE.value == "PHONE"

    def test_account(self):
        assert FraudType.ACCOUNT.value == "ACCOUNT"

    def test_email(self):
        assert FraudType.EMAIL.value == "EMAIL"

    def test_from_string(self):
        assert FraudType("PHONE") == FraudType.PHONE


class TestPhonePattern:
    def test_default(self):
        p = PhonePattern()
        assert p.is_valid is False
        assert p.phone_type == "unknown"
        assert p.warnings == []

    def test_to_dict(self):
        p = PhonePattern(is_valid=True, phone_type="휴대전화", warnings=["경고1"])
        d = p.to_dict()
        assert d["isValid"] is True
        assert d["type"] == "휴대전화"
        assert d["warnings"] == ["경고1"]


class TestAccountPattern:
    def test_default(self):
        a = AccountPattern()
        assert a.is_valid is False
        assert a.bank_code is None

    def test_to_dict_with_bank(self):
        a = AccountPattern(is_valid=True, bank_code="KB", bank_name="국민은행")
        d = a.to_dict()
        assert d["isValid"] is True
        assert d["bankName"] == "국민은행"

    def test_to_dict_without_bank(self):
        a = AccountPattern(is_valid=True)
        d = a.to_dict()
        assert "bankName" not in d
