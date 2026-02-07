from fastapi import APIRouter
from pydantic import BaseModel
import re
import logging
from urllib.parse import urlparse

# 기존 라우터의 함수들 import
from .fraud_router import (
    check_phone_pattern,
    check_account_pattern,
    search_thecheat,
    search_police_cyber,
    search_fss,
    BANK_CODES,
)
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
    # http://, https:// 로 시작하거나
    # 도메인 패턴 (xxx.xxx) 포함
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
    # 한국 전화번호 패턴: 010, 011, 016, 017, 018, 019, 02, 031-064, 070, 080, 1588 등
    phone_patterns = [
        r'^01[016789]\d{7,8}$',  # 휴대전화
        r'^02\d{7,8}$',  # 서울
        r'^0[3-6][1-4]\d{7,8}$',  # 지역번호
        r'^070\d{8}$',  # 인터넷전화
        r'^080\d{7,8}$',  # 수신자부담
        r'^050\d{8,9}$',  # 가상번호
        r'^060\d{7,8}$',  # ARS
        r'^15\d{6}$',  # 대표번호
        r'^16\d{6}$',  # 대표번호
        r'^18\d{6}$',  # 대표번호
    ]

    for pattern in phone_patterns:
        if re.match(pattern, digits_only):
            return 'PHONE', digits_only

    # 4. 계좌번호 감지
    # 일반적으로 10-16자리 숫자
    if 10 <= len(digits_only) <= 16:
        # 전화번호가 아니면 계좌번호로 판단
        # 계좌번호는 보통 전화번호 형식으로 시작하지 않음
        if not digits_only.startswith(('01', '02', '03', '04', '05', '06', '07', '08', '15', '16', '18')):
            return 'ACCOUNT', digits_only
        # 12자리 이상이면 계좌번호일 가능성 높음
        if len(digits_only) >= 12:
            return 'ACCOUNT', digits_only

    # 5. 불분명한 경우 - 길이로 추정
    if len(digits_only) >= 10:
        # 숫자가 10자리 이상이면 계좌번호 우선
        if len(digits_only) <= 11 and digits_only.startswith('0'):
            return 'PHONE', digits_only
        return 'ACCOUNT', digits_only
    elif len(digits_only) >= 8:
        return 'PHONE', digits_only

    return 'UNKNOWN', value


@router.post("/check", response_model=VerifyResponse)
async def unified_verify(request: VerifyRequest):
    """통합 검증 API - URL, 전화번호, 계좌번호 자동 감지 후 검증

    입력값을 자동으로 분석하여 적절한 검증을 수행합니다:
    - URL: 단축 URL 확장, 피싱/악성 사이트 검사
    - 전화번호: 사기 신고 이력 조회
    - 계좌번호: 사기 신고 이력 조회
    """
    try:
        value = request.value.strip()

        if not value:
            return VerifyResponse(
                success=False,
                error="검증할 값을 입력해주세요"
            )

        # 입력 유형 감지
        input_type, normalized_value = detect_input_type(value)

        if input_type == 'UNKNOWN':
            return VerifyResponse(
                success=False,
                error="입력값의 유형을 판별할 수 없습니다. URL, 전화번호, 또는 계좌번호를 입력해주세요."
            )

        # URL 검증
        if input_type == 'URL':
            return await verify_url(normalized_value, value)

        # 전화번호 검증
        elif input_type == 'PHONE':
            return await verify_phone(normalized_value, value)

        # 계좌번호 검증
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
        return VerifyResponse(
            success=False,
            error="올바른 URL 형식이 아닙니다"
        )

    # 단축 URL 감지 (도메인 목록 + 패턴 기반)
    domain_lower = domain.lower()
    is_short_url = any(short_domain in domain_lower for short_domain in SHORT_URL_DOMAINS)

    # 추가 단축 URL 패턴 감지
    if not is_short_url:
        # 도메인에 link, url, short, tiny, gg 등이 포함된 짧은 도메인
        short_patterns = ["link", "url", "short", "tiny", ".gg", ".to", ".cc", ".me"]
        if any(p in domain_lower for p in short_patterns):
            # 경로가 짧으면 (보통 단축 URL은 /abc123 형태)
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

    # 패턴 분석
    suspicious_patterns, risk_score = analyze_url_patterns(url, domain)

    # HTTPS 검사
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

    # Google Safe Browsing
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

    # 위험도 판단
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
    """전화번호 검증"""
    # 패턴 분석
    pattern_analysis = await check_phone_pattern(phone)

    # 더치트 검색
    thecheat_result = await search_thecheat("PHONE", phone)

    # 경찰청 정보
    police_result = await search_police_cyber("PHONE", phone)

    # 상태 결정
    status = "safe"
    total_records = 0

    if thecheat_result.get("found"):
        status = "danger"
        total_records = len(thecheat_result.get("records", []))

    # 메시지 생성
    if status == "danger":
        message = f"사기 신고 이력이 발견되었습니다! ({total_records}건)"
        recommendations = [
            "이 번호와의 거래를 즉시 중단하세요",
            "이미 송금했다면 즉시 경찰(112)에 신고하세요",
            "금융감독원(1332)에 피해 상담을 받으세요"
        ]
    else:
        message = "현재까지 신고된 사기 이력이 없습니다."
        recommendations = [
            "신고 이력이 없다고 해서 100% 안전한 것은 아닙니다",
            "처음 거래하는 상대방에게는 소액 먼저 테스트하세요",
            "의심스러운 경우 경찰청 사이버안전국에 문의하세요"
        ]

    # 패턴 분석 경고 추가
    if pattern_analysis.get("warnings"):
        recommendations = pattern_analysis["warnings"] + recommendations

    # 전화번호 포맷팅
    display_value = phone
    if len(phone) == 11 and phone.startswith("010"):
        display_value = f"{phone[:3]}-{phone[3:7]}-{phone[7:]}"
    elif len(phone) == 10 and phone.startswith("02"):
        display_value = f"{phone[:2]}-{phone[2:6]}-{phone[6:]}"
    elif len(phone) >= 10:
        display_value = f"{phone[:3]}-{phone[3:7]}-{phone[7:]}"

    response_data = {
        "detectedType": "PHONE",
        "detectedTypeLabel": "전화번호",
        "status": status,
        "inputValue": original_value,
        "value": phone,
        "displayValue": display_value,
        "message": message,
        "patternAnalysis": pattern_analysis,
        "totalRecords": total_records,
        "recommendations": recommendations,
        "additionalLinks": [
            {
                "name": "더치트에서 직접 검색",
                "url": thecheat_result.get("searchUrl", "https://thecheat.co.kr"),
                "description": "사기 피해 신고 데이터베이스"
            },
            {
                "name": "경찰청 사이버범죄 신고",
                "url": "https://ecrm.police.go.kr",
                "description": "사이버범죄 신고 및 상담"
            }
        ]
    }

    return VerifyResponse(success=True, data=response_data)


async def verify_account(account: str, original_value: str) -> VerifyResponse:
    """계좌번호 검증"""
    # 패턴 분석
    pattern_analysis = await check_account_pattern(account, None)

    # 더치트 검색
    thecheat_result = await search_thecheat("ACCOUNT", account)

    # 경찰청 정보
    police_result = await search_police_cyber("ACCOUNT", account)

    # 금융감독원 정보
    fss_result = await search_fss(account, None)

    # 상태 결정
    status = "safe"
    total_records = 0

    if thecheat_result.get("found"):
        status = "danger"
        total_records = len(thecheat_result.get("records", []))

    # 메시지 생성
    if status == "danger":
        message = f"사기 신고 이력이 발견되었습니다! ({total_records}건)"
        recommendations = [
            "이 계좌로의 송금을 즉시 중단하세요",
            "이미 송금했다면 즉시 경찰(112)에 신고하세요",
            "금융감독원(1332)에 피해 상담을 받으세요"
        ]
    else:
        message = "현재까지 신고된 사기 이력이 없습니다."
        recommendations = [
            "신고 이력이 없다고 해서 100% 안전한 것은 아닙니다",
            "처음 거래하는 상대방에게는 소액 먼저 테스트하세요",
            "의심스러운 경우 경찰청 사이버안전국에 문의하세요"
        ]

    # 계좌번호 포맷팅
    display_value = account
    if len(account) >= 10:
        display_value = f"{account[:3]}-{account[3:6]}-{account[6:]}"

    response_data = {
        "detectedType": "ACCOUNT",
        "detectedTypeLabel": "계좌번호",
        "status": status,
        "inputValue": original_value,
        "value": account,
        "displayValue": display_value,
        "message": message,
        "patternAnalysis": pattern_analysis,
        "totalRecords": total_records,
        "recommendations": recommendations,
        "additionalLinks": [
            {
                "name": "더치트에서 직접 검색",
                "url": thecheat_result.get("searchUrl", "https://thecheat.co.kr"),
                "description": "사기 피해 신고 데이터베이스"
            },
            {
                "name": "경찰청 사이버범죄 신고",
                "url": "https://ecrm.police.go.kr",
                "description": "사이버범죄 신고 및 상담"
            },
            {
                "name": "금융감독원 보이스피싱 조회",
                "url": "https://www.fss.or.kr/fss/main/sub1sub3.do",
                "description": "피해계좌 조회 및 상담"
            }
        ]
    }

    return VerifyResponse(success=True, data=response_data)
