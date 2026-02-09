from src.domain.fraud.service import (
    analyze_phone_pattern,
    analyze_account_pattern,
    format_phone,
    format_account,
    BANK_CODES,
)


class TestAnalyzePhonePattern:
    def test_valid_mobile(self):
        p = analyze_phone_pattern("01012345678")
        assert p.is_valid is True
        assert p.phone_type == "휴대전화"
        assert p.warnings == []

    def test_voip_warning(self):
        p = analyze_phone_pattern("07012345678")
        assert p.is_valid is True
        assert p.phone_type == "인터넷전화 (VoIP)"
        assert len(p.warnings) == 1
        assert "스팸" in p.warnings[0]

    def test_virtual_number_warning(self):
        p = analyze_phone_pattern("050712345678")
        assert p.is_valid is True
        assert "가상번호" in p.phone_type
        assert any("신원 추적" in w for w in p.warnings)

    def test_paid_call_warning(self):
        p = analyze_phone_pattern("06012345678")
        assert "유료전화" in p.phone_type

    def test_seoul_area(self):
        p = analyze_phone_pattern("0212345678")
        assert p.phone_type == "서울 지역번호"

    def test_representative_number(self):
        p = analyze_phone_pattern("158812345")
        assert p.phone_type == "대표번호"

    def test_regional_number(self):
        p = analyze_phone_pattern("03112345678")
        assert p.phone_type == "지역번호"

    def test_international_82(self):
        p = analyze_phone_pattern("821012345678")
        assert p.phone_type == "국제전화 (한국)"

    def test_too_short(self):
        p = analyze_phone_pattern("1234")
        assert p.is_valid is False
        assert "유효하지 않은 전화번호 길이" in p.warnings

    def test_too_long(self):
        p = analyze_phone_pattern("0101234567890123")
        assert p.is_valid is False

    def test_strips_non_digits(self):
        p = analyze_phone_pattern("010-1234-5678")
        assert p.is_valid is True
        assert p.phone_type == "휴대전화"


class TestAnalyzeAccountPattern:
    def test_valid_account(self):
        a = analyze_account_pattern("1234567890123", None)
        assert a.is_valid is True

    def test_valid_with_bank(self):
        a = analyze_account_pattern("1234567890123", "KB")
        assert a.is_valid is True
        assert a.bank_name == "국민은행"

    def test_unknown_bank_code(self):
        a = analyze_account_pattern("1234567890123", "UNKNOWN_BANK")
        assert a.bank_name is None

    def test_too_short(self):
        a = analyze_account_pattern("12345", None)
        assert a.is_valid is False
        assert "일반적이지 않은 계좌번호 길이" in a.warnings

    def test_too_long(self):
        a = analyze_account_pattern("12345678901234567", None)
        assert a.is_valid is False

    def test_strips_non_digits(self):
        a = analyze_account_pattern("123-456-789012", None)
        assert a.is_valid is True


class TestFormatPhone:
    def test_mobile_010(self):
        assert format_phone("01012345678") == "010-1234-5678"

    def test_seoul_02(self):
        assert format_phone("0212345678") == "02-1234-5678"

    def test_other_10_digit(self):
        assert format_phone("0311234567") == "031-1234-567"

    def test_short_number(self):
        assert format_phone("15881234") == "15881234"


class TestFormatAccount:
    def test_long_account(self):
        assert format_account("1234567890123") == "123-456-7890123"

    def test_short_account(self):
        assert format_account("12345") == "12345"


class TestBankCodes:
    def test_contains_major_banks(self):
        assert "KB" in BANK_CODES
        assert "SHINHAN" in BANK_CODES
        assert "KAKAO" in BANK_CODES
        assert "TOSS" in BANK_CODES

    def test_bank_names_are_korean(self):
        for name in BANK_CODES.values():
            assert len(name) > 0
