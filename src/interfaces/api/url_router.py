from fastapi import APIRouter
from pydantic import BaseModel
from urllib.parse import urlparse, unquote
import httpx
import re
import logging
from src.shared.config import get_settings

router = APIRouter(prefix="/url", tags=["url"])
logger = logging.getLogger(__name__)

# 알려진 단축 URL 서비스 도메인
SHORT_URL_DOMAINS = {
    # 글로벌 서비스
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "mcaf.ee", "su.pr", "yourls.org",
    "cutt.ly", "rb.gy", "shorturl.at", "zpr.io", "rebrand.ly",
    "short.io", "bl.ink", "t2m.io", "clck.ru", "shorten.rest",
    # 한국 서비스
    "han.gl", "vo.la", "me2.do", "url.kr", "링크.kr", "단축.url",
    "link24.kr", "url.asia", "han.gg", "zzb.bz", "kko.to",
    "guni.kr", "s.id", "hoy.kr", "buly.kr", "u.nu",
}

# 위험 도메인 패턴
DANGEROUS_DOMAIN_PATTERNS = [
    r".*-login\..*",
    r".*\.login-.*",
    r".*secure-.*(?<!\.gov)(?<!\.bank)",
    r".*verify-.*",
    r".*account-.*confirm.*",
    r".*update-.*info.*",
    r".*-auth\..*",
]

# 알려진 피싱/스캠 키워드
SCAM_KEYWORDS = {
    "crypto", "bitcoin", "ethereum", "wallet", "airdrop", "giveaway",
    "prize", "winner", "lottery", "invest", "trading", "forex",
    "clickhere", "urgentaction", "suspended", "verify-now",
    "무료코인", "투자수익", "당첨", "긴급", "계정정지", "본인확인"
}


class UrlCheckRequest(BaseModel):
    url: str


class UrlCheckResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None


async def expand_short_url(url: str, max_redirects: int = 10) -> dict:
    """단축 URL을 실제 URL로 확장"""
    redirect_chain = [url]
    current_url = url

    try:
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=10.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; URLChecker/1.0)"}
        ) as client:
            for _ in range(max_redirects):
                try:
                    response = await client.head(current_url, follow_redirects=False)

                    # 리다이렉트 응답인 경우
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if location:
                            # 상대 경로 처리
                            if location.startswith("/"):
                                parsed = urlparse(current_url)
                                location = f"{parsed.scheme}://{parsed.netloc}{location}"

                            current_url = location
                            redirect_chain.append(current_url)
                        else:
                            break
                    else:
                        break
                except httpx.RequestError:
                    # HEAD 요청 실패시 GET으로 재시도
                    try:
                        response = await client.get(current_url, follow_redirects=False)
                        if response.status_code in (301, 302, 303, 307, 308):
                            location = response.headers.get("location")
                            if location:
                                if location.startswith("/"):
                                    parsed = urlparse(current_url)
                                    location = f"{parsed.scheme}://{parsed.netloc}{location}"
                                current_url = location
                                redirect_chain.append(current_url)
                            else:
                                break
                        else:
                            break
                    except:
                        break
    except Exception as e:
        logger.warning(f"URL expansion failed: {e}")

    return {
        "original_url": url,
        "final_url": current_url,
        "redirect_chain": redirect_chain,
        "redirect_count": len(redirect_chain) - 1
    }


async def check_google_safe_browsing(url: str) -> dict | None:
    """Google Safe Browsing API 검사"""
    settings = get_settings()
    api_key = getattr(settings, 'google_safe_browsing_key', None)

    if not api_key:
        return None

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}",
                json={
                    "client": {
                        "clientId": "love-guard",
                        "clientVersion": "1.0.0"
                    },
                    "threatInfo": {
                        "threatTypes": [
                            "MALWARE",
                            "SOCIAL_ENGINEERING",
                            "UNWANTED_SOFTWARE",
                            "POTENTIALLY_HARMFUL_APPLICATION"
                        ],
                        "platformTypes": ["ANY_PLATFORM"],
                        "threatEntryTypes": ["URL"],
                        "threatEntries": [{"url": url}]
                    }
                }
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("matches"):
                    return {
                        "is_dangerous": True,
                        "threats": [m.get("threatType") for m in data["matches"]]
                    }
                return {"is_dangerous": False, "threats": []}
    except Exception as e:
        logger.warning(f"Google Safe Browsing check failed: {e}")

    return None


def analyze_url_patterns(url: str, domain: str) -> tuple[list[str], int]:
    """URL 패턴 분석"""
    suspicious_patterns = []
    risk_score = 0

    # 1. 위험 도메인 패턴 검사
    for pattern in DANGEROUS_DOMAIN_PATTERNS:
        if re.match(pattern, domain, re.IGNORECASE):
            suspicious_patterns.append("피싱 의심 도메인 패턴 감지")
            risk_score += 30
            break

    # 2. 스캠 키워드 검사
    url_lower = url.lower()
    domain_lower = domain.lower()
    for keyword in SCAM_KEYWORDS:
        if keyword in domain_lower:
            suspicious_patterns.append(f'도메인에 의심 키워드 "{keyword}" 포함')
            risk_score += 25
            break
        elif keyword in url_lower:
            suspicious_patterns.append(f'URL에 의심 키워드 "{keyword}" 포함')
            risk_score += 15

    # 3. IP 주소 직접 접속 검사
    if re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", domain):
        suspicious_patterns.append("IP 주소로 직접 접속 (피싱 의심)")
        risk_score += 35

    # 4. 과도한 서브도메인 검사
    subdomain_count = domain.count(".")
    if subdomain_count > 3:
        suspicious_patterns.append(f"과도한 서브도메인 사용 ({subdomain_count}개)")
        risk_score += 20

    # 5. 과도한 하이픈 검사
    if domain.count("-") > 3:
        suspicious_patterns.append("과도한 하이픈 사용")
        risk_score += 15

    # 6. 숫자로 시작하는 도메인
    if re.match(r"^\d", domain.split(".")[0]):
        suspicious_patterns.append("숫자로 시작하는 도메인")
        risk_score += 10

    # 7. 유명 브랜드 유사 도메인 (타이포스쿼팅)
    typosquat_targets = {
        "google": ["g00gle", "googie", "gooogle", "googel"],
        "facebook": ["faceb00k", "facebok", "faceboook"],
        "instagram": ["instagran", "1nstagram", "instgram"],
        "kakao": ["kaka0", "kakoa", "kakaoo"],
        "naver": ["navar", "navor", "n4ver"],
        "samsung": ["samsumg", "samsuung", "samsug"],
    }
    for brand, typos in typosquat_targets.items():
        for typo in typos:
            if typo in domain_lower:
                suspicious_patterns.append(f'"{brand}" 사칭 의심 도메인')
                risk_score += 40
                break

    # 8. 의심스러운 TLD
    suspicious_tlds = [".xyz", ".top", ".click", ".link", ".work", ".date", ".racing", ".download"]
    for tld in suspicious_tlds:
        if domain.endswith(tld):
            suspicious_patterns.append(f'스팸/피싱에 자주 사용되는 도메인 ({tld})')
            risk_score += 15
            break

    # 9. URL 인코딩 악용 검사
    if "%" in url and ("@" in unquote(url) or "?" in domain):
        suspicious_patterns.append("URL 인코딩을 이용한 속임 시도 의심")
        risk_score += 25

    return suspicious_patterns, risk_score


async def perform_url_check(url: str) -> dict:
    """URL 안전성 검사 핵심 로직 — 라우터·종합분석에서 공용 사용"""
    url = url.strip()

    # URL 정규화
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    parsed = urlparse(url)
    domain = parsed.hostname or ""

    # 1. 단축 URL인지 확인하고 확장
    domain_lower = domain.lower()
    is_short_url = any(short_domain in domain_lower for short_domain in SHORT_URL_DOMAINS)

    # 추가 단축 URL 패턴 감지
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

    # 2. 패턴 분석
    suspicious_patterns, risk_score = analyze_url_patterns(url, domain)

    # 3. HTTPS 검사
    is_https = parsed.scheme == "https"
    if not is_https:
        suspicious_patterns.append("HTTPS 미사용 (암호화되지 않은 연결)")
        risk_score += 20

    # 4. 단축 URL 자체도 위험 요소
    if is_short_url:
        suspicious_patterns.append("단축 URL 사용 (최종 목적지 확인됨)")
        risk_score += 10
        if expansion_result and expansion_result["redirect_count"] > 3:
            suspicious_patterns.append(f'과도한 리다이렉트 ({expansion_result["redirect_count"]}회)')
            risk_score += 20

    # 5. Google Safe Browsing 검사
    safe_browsing_result = await check_google_safe_browsing(url)
    if safe_browsing_result:
        if safe_browsing_result.get("is_dangerous"):
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

    # 6. 위험도 판단
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

    original_url = url  # 정규화된 URL
    response_data = {
        "status": status,
        "originalUrl": original_url,
        "finalUrl": url,
        "domain": domain,
        "isHttps": is_https,
        "isShortUrl": is_short_url,
        "riskScore": min(100, risk_score),
        "suspiciousPatterns": suspicious_patterns,
        "message": status_messages[status],
    }

    if expansion_result:
        response_data["expansion"] = {
            "redirectCount": expansion_result["redirect_count"],
            "redirectChain": expansion_result["redirect_chain"]
        }

    if safe_browsing_result:
        response_data["googleSafeBrowsing"] = safe_browsing_result

    return response_data


@router.post("/check", response_model=UrlCheckResponse)
async def check_url_safety(request: UrlCheckRequest):
    """URL 안전성 검사 - 단축 URL 확장, 피싱, 악성코드, 사기 사이트 탐지"""
    try:
        data = await perform_url_check(request.url)
        data["originalUrl"] = request.url  # 원본 입력 값 유지
        return UrlCheckResponse(success=True, data=data)
    except Exception as e:
        logger.error(f"URL check failed: {e}", exc_info=True)
        return UrlCheckResponse(success=False, error=str(e))
