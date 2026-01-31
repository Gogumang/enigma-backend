from fastapi import APIRouter
from pydantic import BaseModel
from urllib.parse import urlparse
import httpx

router = APIRouter(prefix="/url", tags=["url"])


class UrlCheckRequest(BaseModel):
    url: str


class UrlCheckResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None


@router.post("/check", response_model=UrlCheckResponse)
async def check_url_safety(request: UrlCheckRequest):
    """URL 안전성 검사 - 피싱, 악성코드, 사기 사이트 탐지"""
    try:
        url = request.url.strip()

        # URL 정규화
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        try:
            parsed = urlparse(url)
            domain = parsed.hostname or ""
        except Exception:
            return UrlCheckResponse(
                success=False,
                error="올바른 URL 형식이 아닙니다"
            )

        suspicious_patterns = []
        risk_score = 0

        # 기본 패턴 검사
        if "login" in domain or "signin" in domain:
            suspicious_patterns.append("로그인 키워드가 도메인에 포함")
            risk_score += 20

        if "verify" in domain or "secure" in domain:
            suspicious_patterns.append("보안/인증 키워드가 도메인에 포함")
            risk_score += 15

        # IP 주소 직접 접속 검사
        import re
        if re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", domain):
            suspicious_patterns.append("IP 주소로 직접 접속")
            risk_score += 30

        # 과도한 하이픈 검사
        if domain.count("-") > 3:
            suspicious_patterns.append("과도한 하이픈 사용")
            risk_score += 15

        # 의심 키워드 검사
        scam_keywords = ["bit.ly", "tinyurl", "shorturl", "crypto", "invest", "trading", "wallet", "prize", "winner"]
        for keyword in scam_keywords:
            if keyword in domain.lower():
                suspicious_patterns.append(f'의심 키워드 "{keyword}" 포함')
                risk_score += 25

        # HTTPS 검사
        is_https = parsed.scheme == "https"
        if not is_https:
            suspicious_patterns.append("HTTPS 미사용")
            risk_score += 20

        # 외부 URL 검사 API 호출 시도 (Google Safe Browsing, VirusTotal 등)
        external_check_result = None
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Google Safe Browsing API나 VirusTotal API 등 연동 가능
                # 현재는 시뮬레이션
                pass
        except Exception:
            pass

        # 위험도 판단
        status = "safe"
        if risk_score >= 60:
            status = "danger"
        elif risk_score >= 30:
            status = "warning"

        status_messages = {
            "safe": "안전해 보입니다. 하지만 항상 주의하세요.",
            "warning": "의심스러운 패턴이 감지되었습니다. 신중하게 접근하세요.",
            "danger": "피싱이나 악성 사이트일 가능성이 높습니다. 접속을 권장하지 않습니다."
        }

        return UrlCheckResponse(
            success=True,
            data={
                "status": status,
                "url": url,
                "domain": domain,
                "isHttps": is_https,
                "riskScore": min(100, risk_score),
                "suspiciousPatterns": suspicious_patterns,
                "message": status_messages[status],
                "externalCheck": external_check_result
            }
        )

    except Exception as e:
        return UrlCheckResponse(success=False, error=str(e))
