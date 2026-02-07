from fastapi import APIRouter
from pydantic import BaseModel
import httpx
import re
import logging
from bs4 import BeautifulSoup
from urllib.parse import quote

router = APIRouter(prefix="/fraud", tags=["fraud"])
logger = logging.getLogger(__name__)


class FraudCheckRequest(BaseModel):
    type: str  # PHONE, ACCOUNT, EMAIL
    value: str
    bank_code: str | None = None  # 계좌 조회 시 은행 코드


class FraudCheckResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None


# 은행 코드 매핑
BANK_CODES = {
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


async def search_thecheat(query_type: str, value: str) -> dict:
    """더치트(TheCheat) 검색 - 사기 피해 신고 데이터베이스"""
    results = {
        "source": "더치트 (thecheat.co.kr)",
        "found": False,
        "records": [],
        "searchUrl": ""
    }

    try:
        search_url = f"https://thecheat.co.kr/rb/?mod=_search&skind=phone&stxt={quote(value)}"
        if query_type == "ACCOUNT":
            search_url = f"https://thecheat.co.kr/rb/?mod=_search&skind=account&stxt={quote(value)}"

        results["searchUrl"] = search_url

        async with httpx.AsyncClient(
            timeout=15.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            }
        ) as client:
            response = await client.get(search_url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # 검색 결과 파싱
                result_items = soup.select('.result_list li, .search_result .item, table.board_list tr')

                for item in result_items[:10]:  # 최대 10개
                    text = item.get_text(strip=True)
                    if value in text or any(keyword in text for keyword in ["사기", "피해", "신고"]):
                        results["found"] = True
                        results["records"].append({
                            "content": text[:200],
                            "source": "더치트"
                        })

                # 결과가 없다는 문구 확인
                no_result_texts = ["검색결과가 없습니다", "조회된 결과가 없습니다", "등록된 정보가 없습니다"]
                page_text = soup.get_text()
                if any(text in page_text for text in no_result_texts):
                    results["found"] = False
                    results["records"] = []

    except Exception as e:
        logger.warning(f"TheCheat search failed: {e}")
        results["error"] = str(e)

    return results


async def search_police_cyber(query_type: str, value: str) -> dict:
    """경찰청 사이버안전국 - 사이버범죄 신고 검색"""
    results = {
        "source": "경찰청 사이버안전국",
        "found": False,
        "records": [],
        "searchUrl": "https://cyberbureau.police.go.kr"
    }

    # 경찰청 직접 API는 공개되지 않아 안내 링크 제공
    results["guideUrl"] = "https://cyberbureau.police.go.kr/prevention/sub7.jsp?mid=020600"
    results["reportUrl"] = "https://ecrm.police.go.kr/minwon/main"

    return results


async def search_fss(account_number: str, bank_code: str | None) -> dict:
    """금융감독원 - 보이스피싱 피해계좌 조회"""
    results = {
        "source": "금융감독원 (FSS)",
        "found": False,
        "records": [],
        "searchUrl": "https://www.fss.or.kr/fss/main/sub1sub3.do"
    }

    try:
        # 금융감독원 통합신고센터 안내
        results["guideUrl"] = "https://www.fss.or.kr/fss/main/sub1sub3.do"
        results["hotline"] = "1332 (금융감독원 콜센터)"

        # 실제 피해계좌 여부는 은행 앱에서 확인 가능
        results["checkMethods"] = [
            "해당 은행 앱/홈페이지에서 '피해계좌 조회' 메뉴 이용",
            "금융감독원 1332 전화 상담",
            "경찰청 사이버범죄 신고시스템 이용"
        ]

    except Exception as e:
        logger.warning(f"FSS search failed: {e}")

    return results


async def check_phone_pattern(phone: str) -> dict:
    """전화번호 패턴 분석"""
    analysis = {
        "isValid": False,
        "type": "unknown",
        "warnings": []
    }

    # 숫자만 추출
    digits = re.sub(r'\D', '', phone)

    # 유효성 검사
    if len(digits) < 9 or len(digits) > 12:
        analysis["warnings"].append("유효하지 않은 전화번호 길이")
        return analysis

    analysis["isValid"] = True

    # 번호 유형 판별
    if digits.startswith("02"):
        analysis["type"] = "서울 지역번호"
    elif digits.startswith("010"):
        analysis["type"] = "휴대전화"
    elif digits.startswith("070"):
        analysis["type"] = "인터넷전화 (VoIP)"
        analysis["warnings"].append("인터넷전화는 스팸/스캠에 자주 사용됩니다")
    elif digits.startswith("050"):
        analysis["type"] = "안심번호/가상번호"
        analysis["warnings"].append("가상번호는 신원 추적이 어렵습니다")
    elif digits.startswith("060"):
        analysis["type"] = "유료전화 (ARS)"
        analysis["warnings"].append("유료전화 - 요금이 부과될 수 있습니다")
    elif digits.startswith("080"):
        analysis["type"] = "수신자부담전화"
    elif digits.startswith("15") or digits.startswith("16") or digits.startswith("18"):
        analysis["type"] = "대표번호"
    elif digits.startswith(("031", "032", "033", "041", "042", "043", "044",
                            "051", "052", "053", "054", "055", "061", "062", "063", "064")):
        analysis["type"] = "지역번호"
    elif digits.startswith("82"):
        analysis["type"] = "국제전화 (한국)"
        analysis["warnings"].append("국제전화 형식입니다")
    elif digits.startswith("+") or len(digits) > 11:
        analysis["type"] = "국제전화"
        analysis["warnings"].append("해외 번호로 의심됩니다")

    return analysis


async def check_account_pattern(account: str, bank_code: str | None) -> dict:
    """계좌번호 패턴 분석"""
    analysis = {
        "isValid": False,
        "bank": bank_code,
        "warnings": []
    }

    # 숫자와 하이픈만 추출
    digits = re.sub(r'\D', '', account)

    # 유효성 검사 (은행별 계좌번호 길이 다름, 일반적으로 10-14자리)
    if len(digits) < 10 or len(digits) > 16:
        analysis["warnings"].append("일반적이지 않은 계좌번호 길이")
    else:
        analysis["isValid"] = True

    # 은행명 추가
    if bank_code and bank_code in BANK_CODES:
        analysis["bankName"] = BANK_CODES[bank_code]

    return analysis


@router.post("/check", response_model=FraudCheckResponse)
async def check_fraud(request: FraudCheckRequest):
    """전화번호, 계좌번호의 사기 이력 조회

    여러 소스에서 사기 신고 이력을 검색합니다:
    - 더치트 (thecheat.co.kr): 사기 피해 신고 커뮤니티
    - 경찰청 사이버안전국: 사이버범죄 신고/조회
    - 금융감독원: 보이스피싱 피해계좌 조회
    """
    try:
        check_type = request.type.upper()
        value = request.value.replace("-", "").replace(" ", "").strip()

        if check_type not in ["PHONE", "ACCOUNT"]:
            return FraudCheckResponse(
                success=False,
                error="지원하는 조회 유형: PHONE (전화번호), ACCOUNT (계좌번호)"
            )

        # 기본 응답 구조
        response_data = {
            "status": "safe",
            "type": check_type,
            "value": value,
            "displayValue": request.value,  # 원본 형식 유지
            "sources": [],
            "totalRecords": 0,
            "message": "",
            "recommendations": []
        }

        # 1. 패턴 분석
        if check_type == "PHONE":
            pattern_analysis = await check_phone_pattern(value)
            response_data["patternAnalysis"] = pattern_analysis

            if pattern_analysis.get("warnings"):
                response_data["recommendations"].extend(pattern_analysis["warnings"])

        elif check_type == "ACCOUNT":
            pattern_analysis = await check_account_pattern(value, request.bank_code)
            response_data["patternAnalysis"] = pattern_analysis

            if request.bank_code:
                response_data["bank"] = BANK_CODES.get(request.bank_code, request.bank_code)

        # 2. 더치트 검색
        thecheat_result = await search_thecheat(check_type, value)
        response_data["sources"].append(thecheat_result)

        if thecheat_result.get("found"):
            response_data["status"] = "danger"
            response_data["totalRecords"] += len(thecheat_result.get("records", []))

        # 3. 경찰청 정보
        police_result = await search_police_cyber(check_type, value)
        response_data["sources"].append(police_result)

        # 4. 계좌번호인 경우 금융감독원 정보 추가
        if check_type == "ACCOUNT":
            fss_result = await search_fss(value, request.bank_code)
            response_data["sources"].append(fss_result)

        # 5. 최종 메시지 생성
        if response_data["status"] == "danger":
            response_data["message"] = f"⚠️ 사기 신고 이력이 발견되었습니다! ({response_data['totalRecords']}건)"
            response_data["recommendations"].extend([
                "이 번호/계좌와의 거래를 즉시 중단하세요",
                "이미 송금했다면 즉시 경찰(112)에 신고하세요",
                "금융감독원(1332)에 피해 상담을 받으세요"
            ])
        else:
            response_data["message"] = "현재까지 신고된 사기 이력이 없습니다."
            response_data["recommendations"].extend([
                "신고 이력이 없다고 해서 100% 안전한 것은 아닙니다",
                "처음 거래하는 상대방에게는 소액 먼저 테스트하세요",
                "의심스러운 경우 경찰청 사이버안전국에 문의하세요"
            ])

        # 6. 추가 확인 링크
        response_data["additionalLinks"] = [
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
            },
            {
                "name": "국가정보원 피싱사이트 신고",
                "url": "https://www.nis.go.kr",
                "description": "피싱사이트 신고"
            }
        ]

        return FraudCheckResponse(success=True, data=response_data)

    except Exception as e:
        logger.error(f"Fraud check failed: {e}", exc_info=True)
        return FraudCheckResponse(success=False, error=str(e))


