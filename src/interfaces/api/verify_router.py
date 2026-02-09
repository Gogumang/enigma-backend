from fastapi import APIRouter
from pydantic import BaseModel
import re
import logging
from urllib.parse import urlparse

from src.domain.fraud.service import (
    format_phone,
    format_account,
)
from .fraud_router import get_fraud_use_case
from .url_router import (
    expand_short_url,
    analyze_url_patterns,
    check_google_safe_browsing,
    SHORT_URL_DOMAINS,
)

router = APIRouter(prefix="/verify", tags=["verify"])
logger = logging.getLogger(__name__)


class VerifyRequest(BaseModel):
    value: str


class VerifyResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None


def detect_input_type(value: str) -> tuple[str, str]:
    """입력값의 유형을 자동 감지

    Returns:
        tuple: (type, normalized_value)
        type: 'URL', 'PHONE', 'ACCOUNT', 'UNKNOWN'
    """
    value = value.strip()

    # 1. URL 감지
    if value.startswith(('http://', 'https://')):
        return 'URL', value

    # 단축 URL 도메인 체크
    for short_domain in SHORT_URL_DOMAINS:
        if value.startswith(short_domain) or f".{short_domain}" in value:
            return 'URL', f"https://{value}" if not value.startswith('http') else value

    # 일반 도메인 패턴 (xxx.com, xxx.co.kr 등)
    domain_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9-]*\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?(?:/.*)?$'
    if re.match(domain_pattern, value):
        return 'URL', f"https://{value}"

    # www로 시작
    if value.lower().startswith('www.'):
        return 'URL', f"https://{value}"

    # 2. 숫자와 하이픈만 추출
    digits_only = re.sub(r'[^\d]', '', value)

    # 3. 전화번호 감지
    phone_patterns = [
        r'^01[016789]\d{7,8}$',
        r'^02\d{7,8}$',
        r'^0[3-6][1-4]\d{7,8}$',
        r'^070\d{8}$',
        r'^080\d{7,8}$',
        r'^050\d{8,9}$',
        r'^060\d{7,8}$',
        r'^15\d{6}$',
        r'^16\d{6}$',
        r'^18\d{6}$',
    ]

    for pattern in phone_patterns:
        if re.match(pattern, digits_only):
            return 'PHONE', digits_only

    # 4. 계좌번호 감지
    if 10 <= len(digits_only) <= 16:
        if not digits_only.startswith(('01', '02', '03', '04', '05', '06', '07', '08', '15', '16', '18')):
            return 'ACCOUNT', digits_only
        if len(digits_only) >= 12:
            return 'ACCOUNT', digits_only

    # 5. 불분명한 경우 - 길이로 추정
    if len(digits_only) >= 10:
        if len(digits_only) <= 11 and digits_only.startswith('0'):
            return 'PHONE', digits_only
        return 'ACCOUNT', digits_only
    elif len(digits_only) >= 8:
        return 'PHONE', digits_only

    return 'UNKNOWN', value


@router.post("/check", response_model=VerifyResponse)
async def unified_verify(request: VerifyRequest):
    """통합 검증 API - URL, 전화번호, 계좌번호 자동 감지 후 검증"""
    try:
        value = request.value.strip()

        if not value:
            return VerifyResponse(success=False, error="검증할 값을 입력해주세요")

        input_type, normalized_value = detect_input_type(value)

        if input_type == 'UNKNOWN':
            return VerifyResponse(
                success=False,
                error="입력값의 유형을 판별할 수 없습니다. URL, 전화번호, 또는 계좌번호를 입력해주세요."
            )

        if input_type == 'URL':
            return await verify_url(normalized_value, value)
        elif input_type == 'PHONE':
            return await verify_phone(normalized_value, value)
        elif input_type == 'ACCOUNT':
            return await verify_account(normalized_value, value)

    except Exception as e:
        logger.error(f"Unified verify failed: {e}", exc_info=True)
        return VerifyResponse(success=False, error=str(e))


@router.post("/url", response_model=VerifyResponse)
async def verify_url_endpoint(request: VerifyRequest):
    """URL 검증 전용 엔드포인트"""
    try:
        value = request.value.strip()
        if not value:
            return VerifyResponse(success=False, error="URL을 입력해주세요")
        if not value.startswith(("http://", "https://")):
            value = f"https://{value}"
        return await verify_url(value, request.value.strip())
    except Exception as e:
        logger.error(f"URL verify failed: {e}", exc_info=True)
        return VerifyResponse(success=False, error=str(e))


@router.post("/phone", response_model=VerifyResponse)
async def verify_phone_endpoint(request: VerifyRequest):
    """전화번호 검증 전용 엔드포인트"""
    try:
        value = request.value.strip()
        if not value:
            return VerifyResponse(success=False, error="전화번호를 입력해주세요")
        digits = re.sub(r'[^\d]', '', value)
        return await verify_phone(digits, value)
    except Exception as e:
        logger.error(f"Phone verify failed: {e}", exc_info=True)
        return VerifyResponse(success=False, error=str(e))


@router.post("/account", response_model=VerifyResponse)
async def verify_account_endpoint(request: VerifyRequest):
    """계좌번호 검증 전용 엔드포인트"""
    try:
        value = request.value.strip()
        if not value:
            return VerifyResponse(success=False, error="계좌번호를 입력해주세요")
        digits = re.sub(r'[^\d]', '', value)
        return await verify_account(digits, value)
    except Exception as e:
        logger.error(f"Account verify failed: {e}", exc_info=True)
        return VerifyResponse(success=False, error=str(e))


async def verify_url(url: str, original_value: str) -> VerifyResponse:
    """URL 검증"""
    try:
        parsed = urlparse(url)
        domain = parsed.hostname or ""
    except Exception:
        return VerifyResponse(success=False, error="올바른 URL 형식이 아닙니다")

    domain_lower = domain.lower()
    is_short_url = any(short_domain in domain_lower for short_domain in SHORT_URL_DOMAINS)

    if not is_short_url:
        short_patterns = ["link", "url", "short", "tiny", ".gg", ".to", ".cc", ".me"]
        if any(p in domain_lower for p in short_patterns):
            path = parsed.path.strip("/")
            if len(path) > 0 and len(path) <= 15 and "/" not in path:
                is_short_url = True

    expansion_result = None

    if is_short_url:
        expansion_result = await expand_short_url(url)
        if expansion_result["final_url"] != url:
            url = expansion_result["final_url"]
            try:
                parsed = urlparse(url)
                domain = parsed.hostname or ""
            except:
                pass

    suspicious_patterns, risk_score = analyze_url_patterns(url, domain)

    is_https = parsed.scheme == "https"
    if not is_https:
        suspicious_patterns.append("HTTPS 미사용 (암호화되지 않은 연결)")
        risk_score += 20

    if is_short_url:
        suspicious_patterns.append("단축 URL 사용 (최종 목적지 확인됨)")
        risk_score += 10
        if expansion_result and expansion_result["redirect_count"] > 3:
            suspicious_patterns.append(f'과도한 리다이렉트 ({expansion_result["redirect_count"]}회)')
            risk_score += 20

    safe_browsing_result = await check_google_safe_browsing(url)
    if safe_browsing_result and safe_browsing_result.get("is_dangerous"):
        threats = safe_browsing_result.get("threats", [])
        threat_names = {
            "MALWARE": "악성코드",
            "SOCIAL_ENGINEERING": "소셜 엔지니어링(피싱)",
            "UNWANTED_SOFTWARE": "원치 않는 소프트웨어",
            "POTENTIALLY_HARMFUL_APPLICATION": "잠재적 유해 앱"
        }
        for threat in threats:
            suspicious_patterns.append(f'Google 경고: {threat_names.get(threat, threat)}')
        risk_score += 50

    status = "safe"
    if risk_score >= 60:
        status = "danger"
    elif risk_score >= 30:
        status = "warning"

    status_messages = {
        "safe": "안전해 보입니다. 하지만 개인정보 입력 시 항상 주의하세요.",
        "warning": "의심스러운 패턴이 감지되었습니다. 신중하게 접근하세요.",
        "danger": "위험한 사이트일 가능성이 높습니다. 접속을 권장하지 않습니다!"
    }

    response_data = {
        "detectedType": "URL",
        "detectedTypeLabel": "URL",
        "status": status,
        "inputValue": original_value,
        "originalUrl": original_value,
        "finalUrl": url,
        "domain": domain,
        "isHttps": is_https,
        "isShortUrl": is_short_url,
        "riskScore": min(100, risk_score),
        "suspiciousPatterns": suspicious_patterns,
        "message": status_messages[status],
        "recommendations": [
            "개인정보 입력 전 URL을 다시 확인하세요",
            "의심스러운 링크는 클릭하지 마세요",
            "공식 앱이나 북마크를 통해 접속하세요"
        ] if status != "safe" else [],
    }

    if expansion_result:
        response_data["expansion"] = {
            "redirectCount": expansion_result["redirect_count"],
            "redirectChain": expansion_result["redirect_chain"]
        }

    return VerifyResponse(success=True, data=response_data)


async def verify_phone(phone: str, original_value: str) -> VerifyResponse:
    """전화번호 검증 — UseCase 위임"""
    use_case = get_fraud_use_case()
    result = await use_case.execute("PHONE", phone)

    data = result.to_dict()
    data["detectedType"] = "PHONE"
    data["detectedTypeLabel"] = "전화번호"
    data["inputValue"] = original_value
    data["displayValue"] = format_phone(phone)

    return VerifyResponse(success=True, data=data)


async def verify_account(account: str, original_value: str) -> VerifyResponse:
    """계좌번호 검증 — UseCase 위임"""
    use_case = get_fraud_use_case()
    result = await use_case.execute("ACCOUNT", account)

    data = result.to_dict()
    data["detectedType"] = "ACCOUNT"
    data["detectedTypeLabel"] = "계좌번호"
    data["inputValue"] = original_value
    data["displayValue"] = format_account(account)

    return VerifyResponse(success=True, data=data)
