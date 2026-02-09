import re

from .value_objects import PhonePattern, AccountPattern

BANK_CODES: dict[str, str] = {
    "KB": "국민은행",
    "SHINHAN": "신한은행",
    "WOORI": "우리은행",
    "HANA": "하나은행",
    "NH": "농협은행",
    "IBK": "기업은행",
    "SC": "SC제일은행",
    "CITI": "씨티은행",
    "KAKAO": "카카오뱅크",
    "TOSS": "토스뱅크",
    "KBANK": "케이뱅크",
    "POST": "우체국",
    "SUHYUP": "수협은행",
    "BUSAN": "부산은행",
    "DAEGU": "대구은행",
    "GWANGJU": "광주은행",
    "JEJU": "제주은행",
    "JEONBUK": "전북은행",
    "KYONGNAM": "경남은행",
    "SAEMAUL": "새마을금고",
    "CREDIT": "신협",
}


def analyze_phone_pattern(phone: str) -> PhonePattern:
    """전화번호 패턴 분석 — 순수 도메인 로직"""
    pattern = PhonePattern()
    digits = re.sub(r"\D", "", phone)

    if len(digits) < 9 or len(digits) > 12:
        pattern.warnings.append("유효하지 않은 전화번호 길이")
        return pattern

    pattern.is_valid = True

    prefix_map: list[tuple[tuple[str, ...], str, str | None]] = [
        (("02",), "서울 지역번호", None),
        (("010",), "휴대전화", None),
        (("070",), "인터넷전화 (VoIP)", "인터넷전화는 스팸/스캠에 자주 사용됩니다"),
        (("050",), "안심번호/가상번호", "가상번호는 신원 추적이 어렵습니다"),
        (("060",), "유료전화 (ARS)", "유료전화 - 요금이 부과될 수 있습니다"),
        (("080",), "수신자부담전화", None),
        (("15", "16", "18"), "대표번호", None),
        (
            ("031", "032", "033", "041", "042", "043", "044",
             "051", "052", "053", "054", "055", "061", "062", "063", "064"),
            "지역번호",
            None,
        ),
        (("82",), "국제전화 (한국)", "국제전화 형식입니다"),
    ]

    for prefixes, phone_type, warning in prefix_map:
        if digits.startswith(prefixes):
            pattern.phone_type = phone_type
            if warning:
                pattern.warnings.append(warning)
            return pattern

    # fallback
    if digits.startswith("+") or len(digits) > 11:
        pattern.phone_type = "국제전화"
        pattern.warnings.append("해외 번호로 의심됩니다")

    return pattern


def analyze_account_pattern(account: str, bank_code: str | None) -> AccountPattern:
    """계좌번호 패턴 분석 — 순수 도메인 로직"""
    digits = re.sub(r"\D", "", account)
    pattern = AccountPattern(bank_code=bank_code)

    if len(digits) < 10 or len(digits) > 16:
        pattern.warnings.append("일반적이지 않은 계좌번호 길이")
    else:
        pattern.is_valid = True

    if bank_code and bank_code in BANK_CODES:
        pattern.bank_name = BANK_CODES[bank_code]

    return pattern


def format_phone(phone: str) -> str:
    """전화번호 포맷팅"""
    if len(phone) == 11 and phone.startswith("010"):
        return f"{phone[:3]}-{phone[3:7]}-{phone[7:]}"
    if len(phone) == 10 and phone.startswith("02"):
        return f"{phone[:2]}-{phone[2:6]}-{phone[6:]}"
    if len(phone) >= 10:
        return f"{phone[:3]}-{phone[3:7]}-{phone[7:]}"
    return phone


def format_account(account: str) -> str:
    """계좌번호 포맷팅"""
    if len(account) >= 10:
        return f"{account[:3]}-{account[3:6]}-{account[6:]}"
    return account
