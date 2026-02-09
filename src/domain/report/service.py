"""
신고 도우미 도메인 서비스
- 스캠 유형 분류 (가중치 점수)
- 위험도 판단
- 은행 지급정지 번호
- 유형별 신고 절차
- 신고 기관 정보
- 증거 요약
"""
from .value_objects import ScamType, DangerLevel
from .entity import EmergencyAction, ReportingStep, AgencyInfo, EvidenceSummary


# ==================== 은행 지급정지 전화번호 ====================

BANK_EMERGENCY_PHONES: dict[str, str] = {
    "국민은행": "1588-9999",
    "신한은행": "1577-8000",
    "우리은행": "1588-5000",
    "하나은행": "1599-1111",
    "농협은행": "1661-3000",
    "기업은행": "1566-2566",
    "SC제일은행": "1588-1599",
    "카카오뱅크": "1599-3333",
    "토스뱅크": "1661-7654",
    "케이뱅크": "1522-1000",
    "우체국": "1588-1900",
    "수협은행": "1588-1515",
    "새마을금고": "1599-9000",
    "신협": "1566-6000",
}


def classify_scam_type(analysis_results: dict) -> ScamType:
    """분석 결과로부터 스캠 유형 분류 (가중치 점수 기반)"""
    scores: dict[ScamType, int] = {
        ScamType.ROMANCE: 0,
        ScamType.VOICE_PHISHING: 0,
        ScamType.INVESTMENT: 0,
        ScamType.PHISHING: 0,
    }

    # 딥페이크 결과
    deepfake = analysis_results.get("deepfake") or {}
    if deepfake.get("isDeepfake") or (deepfake.get("confidence", 0) > 60):
        scores[ScamType.ROMANCE] += 4

    # 프로필 결과
    profile = analysis_results.get("profile") or {}
    if profile.get("isStolen") or profile.get("stolenConfidence", 0) > 50:
        scores[ScamType.ROMANCE] += 2

    # 사기 이력 결과
    fraud = analysis_results.get("fraud") or {}
    if fraud.get("status") == "danger":
        fraud_type = fraud.get("type", "").upper()
        if fraud_type == "PHONE":
            scores[ScamType.VOICE_PHISHING] += 3
        elif fraud_type == "ACCOUNT":
            scores[ScamType.VOICE_PHISHING] += 2
            scores[ScamType.INVESTMENT] += 1

    # 채팅 분석 결과
    chat = analysis_results.get("chat") or {}
    detected_patterns = chat.get("detectedPatterns", [])
    for pattern in detected_patterns:
        p = pattern.lower() if isinstance(pattern, str) else ""
        if any(kw in p for kw in ["love_bombing", "사랑", "애정", "romance"]):
            scores[ScamType.ROMANCE] += 2
        if any(kw in p for kw in ["financial", "금전", "투자", "investment"]):
            scores[ScamType.INVESTMENT] += 3
        if any(kw in p for kw in ["urgency", "긴급", "급해"]):
            scores[ScamType.VOICE_PHISHING] += 1

    # URL 분석 결과
    url = analysis_results.get("url") or {}
    if url.get("isMalicious") or url.get("riskLevel") == "danger":
        scores[ScamType.PHISHING] += 3
    if url.get("isPhishing"):
        scores[ScamType.PHISHING] += 2

    # 최고 점수 유형 선택
    max_score = max(scores.values())
    if max_score == 0:
        return ScamType.UNKNOWN

    for scam_type, score in scores.items():
        if score == max_score:
            return scam_type

    return ScamType.UNKNOWN


def determine_danger_level(analysis_results: dict) -> DangerLevel:
    """분석 결과로부터 위험 수준 판단"""
    # 종합 점수
    overall_score = analysis_results.get("overallScore", 0)
    if isinstance(overall_score, (int, float)):
        if overall_score >= 80:
            return DangerLevel.CRITICAL
        if overall_score >= 60:
            return DangerLevel.HIGH
        if overall_score >= 40:
            return DangerLevel.MEDIUM
        return DangerLevel.LOW

    # 개별 점수 합산
    scores = [
        analysis_results.get("deepfake", {}).get("confidence", 0),
        analysis_results.get("chat", {}).get("riskScore", 0),
        analysis_results.get("fraud", {}).get("totalRecords", 0) * 20,
    ]
    avg = sum(scores) / max(len(scores), 1)
    if avg >= 70:
        return DangerLevel.CRITICAL
    if avg >= 50:
        return DangerLevel.HIGH
    if avg >= 30:
        return DangerLevel.MEDIUM
    return DangerLevel.LOW


def build_emergency_actions(
    scam_type: ScamType,
    danger_level: DangerLevel,
) -> list[EmergencyAction]:
    """긴급 조치 목록 생성"""
    actions: list[EmergencyAction] = []

    # 위험도가 높으면 지급정지 안내
    if danger_level in (DangerLevel.HIGH, DangerLevel.CRITICAL):
        actions.append(EmergencyAction(
            action="즉시 거래 은행에 지급정지 신청",
            contact="아래 은행별 번호 참조",
            is_urgent=True,
            deadline_hours=30,
            golden_time_warning="송금 후 30분 이내 지급정지 신청 시 환급 가능성이 가장 높습니다.",
        ))

        # 은행별 번호 추가
        for bank, phone in list(BANK_EMERGENCY_PHONES.items())[:5]:
            actions.append(EmergencyAction(
                action=f"{bank} 지급정지",
                contact=phone,
                is_urgent=True,
            ))

    # 유형별 긴급 조치
    if scam_type == ScamType.VOICE_PHISHING:
        actions.append(EmergencyAction(
            action="즉시 전화 끊기 및 발신번호 차단",
            contact="",
            is_urgent=True,
        ))
        actions.append(EmergencyAction(
            action="경찰 긴급신고",
            contact="112",
            is_urgent=True,
        ))
    elif scam_type == ScamType.PHISHING:
        actions.append(EmergencyAction(
            action="의심 URL 접속 차단 및 비밀번호 즉시 변경",
            contact="",
            is_urgent=True,
        ))

    return actions


def build_reporting_steps(scam_type: ScamType) -> list[ReportingStep]:
    """유형별 신고 절차 생성"""
    if scam_type == ScamType.ROMANCE:
        return [
            ReportingStep(
                step=1, title="증거 확보",
                description="대화 내용 스크린샷, 송금 내역, 상대방 프로필 사진 등 모든 증거를 저장하세요.",
                tip="카카오톡/메신저의 '대화 내보내기' 기능을 활용하세요.",
            ),
            ReportingStep(
                step=2, title="은행 지급정지 신청",
                description="송금한 은행에 즉시 전화하여 지급정지를 신청하세요.",
                tip="30분 이내 신청 시 환급 가능성이 높습니다.",
            ),
            ReportingStep(
                step=3, title="경찰청 사이버수사국 신고",
                description="사이버범죄 신고시스템에서 온라인으로 신고하세요.",
                url="https://ecrm.police.go.kr",
                tip="증거 파일을 첨부하면 수사에 도움이 됩니다.",
            ),
            ReportingStep(
                step=4, title="금융감독원 상담",
                description="금융감독원에 피해 상담 및 추가 조치를 문의하세요.",
                url="https://www.fss.or.kr",
                tip="1332로 전화 상담도 가능합니다.",
            ),
            ReportingStep(
                step=5, title="추가 피해 방지",
                description="SNS 계정 비밀번호 변경, 의심 계정 차단, 주변 지인에게 알리세요.",
            ),
        ]
    elif scam_type == ScamType.VOICE_PHISHING:
        return [
            ReportingStep(
                step=1, title="즉시 전화 끊기",
                description="의심 전화를 즉시 끊고, 절대 다시 전화하지 마세요.",
                tip="공공기관은 전화로 금전을 요구하지 않습니다.",
            ),
            ReportingStep(
                step=2, title="은행 지급정지 신청",
                description="이미 송금했다면 즉시 은행에 지급정지를 신청하세요.",
                tip="골든타임: 송금 후 30분 이내 지급정지 신청이 중요합니다.",
            ),
            ReportingStep(
                step=3, title="112 신고",
                description="경찰에 즉시 전화 신고하세요.",
                tip="보이스피싱 전담 수사팀으로 연결됩니다.",
            ),
            ReportingStep(
                step=4, title="경찰청 사이버수사국 온라인 신고",
                description="추가 증거와 함께 온라인 신고를 접수하세요.",
                url="https://ecrm.police.go.kr",
            ),
            ReportingStep(
                step=5, title="악성앱 삭제 및 점검",
                description="스마트폰에 설치된 의심 앱을 삭제하고, 백신으로 전체 검사하세요.",
                tip="원격제어 앱(TeamViewer 등)이 설치되어 있다면 즉시 삭제하세요.",
            ),
        ]
    elif scam_type == ScamType.INVESTMENT:
        return [
            ReportingStep(
                step=1, title="추가 투자 즉시 중단",
                description="더 이상의 금전 송금을 즉시 중단하세요.",
                tip="'원금 회수를 위한 추가 입금' 요구는 100% 사기입니다.",
            ),
            ReportingStep(
                step=2, title="증거 수집",
                description="투자 약정서, 대화 내역, 송금 기록, 앱 스크린샷 등을 저장하세요.",
                tip="SNS 광고나 홍보 게시물도 캡처해두세요.",
            ),
            ReportingStep(
                step=3, title="은행 지급정지 신청",
                description="송금한 은행에 지급정지를 신청하세요.",
            ),
            ReportingStep(
                step=4, title="경찰청 사이버수사국 신고",
                description="수집한 증거와 함께 온라인 신고를 접수하세요.",
                url="https://ecrm.police.go.kr",
            ),
            ReportingStep(
                step=5, title="금융감독원 신고",
                description="금융감독원에 불법 투자업체 신고 및 피해 상담을 받으세요.",
                url="https://www.fss.or.kr",
                tip="금감원 불법금융신고센터: 1332",
            ),
        ]
    elif scam_type == ScamType.PHISHING:
        return [
            ReportingStep(
                step=1, title="의심 사이트 접속 차단",
                description="해당 URL에 다시 접속하지 말고, 북마크도 삭제하세요.",
                tip="브라우저 방문 기록에서도 삭제하세요.",
            ),
            ReportingStep(
                step=2, title="비밀번호 즉시 변경",
                description="해당 사이트에 입력한 계정의 비밀번호를 즉시 변경하세요. 같은 비밀번호를 쓰는 다른 사이트도 모두 변경하세요.",
                tip="2단계 인증(2FA)을 활성화하세요.",
            ),
            ReportingStep(
                step=3, title="KISA 인터넷침해대응센터 신고",
                description="한국인터넷진흥원에 피싱 사이트를 신고하세요.",
                url="https://www.krcert.or.kr",
                tip="118로 전화 상담도 가능합니다.",
            ),
            ReportingStep(
                step=4, title="경찰청 사이버수사국 신고",
                description="개인정보 유출이 의심되면 경찰에도 신고하세요.",
                url="https://ecrm.police.go.kr",
            ),
            ReportingStep(
                step=5, title="금융 계좌 확인",
                description="금융정보를 입력했다면 은행에 연락하여 카드 정지 및 계좌 모니터링을 요청하세요.",
            ),
        ]
    else:
        # UNKNOWN
        return [
            ReportingStep(
                step=1, title="증거 확보",
                description="관련 대화, 스크린샷, 송금 내역 등 모든 증거를 저장하세요.",
            ),
            ReportingStep(
                step=2, title="경찰청 사이버수사국 신고",
                description="사이버범죄 신고시스템에서 온라인으로 신고하세요.",
                url="https://ecrm.police.go.kr",
            ),
            ReportingStep(
                step=3, title="금융감독원 상담",
                description="금전 피해가 있다면 금융감독원에 상담하세요.",
                url="https://www.fss.or.kr",
                tip="전화: 1332",
            ),
        ]


def build_agencies(scam_type: ScamType) -> list[AgencyInfo]:
    """신고 기관 정보"""
    agencies = [
        AgencyInfo(
            name="경찰청 사이버수사국",
            phone="182",
            url="https://ecrm.police.go.kr",
        ),
        AgencyInfo(
            name="금융감독원",
            phone="1332",
            url="https://www.fss.or.kr",
        ),
    ]

    if scam_type == ScamType.PHISHING:
        agencies.append(AgencyInfo(
            name="KISA 인터넷침해대응센터",
            phone="118",
            url="https://www.krcert.or.kr",
        ))

    if scam_type in (ScamType.VOICE_PHISHING, ScamType.ROMANCE, ScamType.INVESTMENT):
        agencies.append(AgencyInfo(
            name="검찰청 보이스피싱 신고",
            phone="1301",
        ))

    return agencies


def build_evidence_summary(analysis_results: dict) -> list[EvidenceSummary]:
    """분석 결과를 증거 요약으로 변환"""
    summaries: list[EvidenceSummary] = []

    # 딥페이크
    deepfake = analysis_results.get("deepfake") or {}
    confidence = deepfake.get("confidence", 0)
    if deepfake.get("isDeepfake") or confidence > 50:
        summaries.append(EvidenceSummary(
            category="deepfake",
            risk_level="danger",
            summary=f"딥페이크 탐지됨 ({confidence:.1f}%)",
        ))
    elif confidence > 0:
        summaries.append(EvidenceSummary(
            category="deepfake",
            risk_level="safe",
            summary=f"딥페이크 미탐지 (신뢰도 {100 - confidence:.1f}%)",
        ))

    # 채팅 분석
    chat = analysis_results.get("chat") or {}
    risk_score = chat.get("riskScore", 0)
    if risk_score > 60:
        patterns = chat.get("detectedPatterns", [])
        pattern_str = ", ".join(patterns[:3]) if patterns else "위험 패턴"
        summaries.append(EvidenceSummary(
            category="chat",
            risk_level="danger",
            summary=f"채팅 위험도 {risk_score}점 - {pattern_str}",
        ))
    elif risk_score > 30:
        summaries.append(EvidenceSummary(
            category="chat",
            risk_level="warning",
            summary=f"채팅 위험도 {risk_score}점 - 주의 필요",
        ))

    # 사기 이력
    fraud = analysis_results.get("fraud") or {}
    if fraud.get("status") == "danger":
        total = fraud.get("totalRecords", 0)
        summaries.append(EvidenceSummary(
            category="fraud",
            risk_level="danger",
            summary=f"사기 신고 이력 {total}건 발견",
        ))

    # 프로필
    profile = analysis_results.get("profile") or {}
    if profile.get("isStolen") or profile.get("stolenConfidence", 0) > 50:
        summaries.append(EvidenceSummary(
            category="profile",
            risk_level="danger",
            summary="프로필 사진 도용 의심",
        ))

    # URL
    url = analysis_results.get("url") or {}
    if url.get("isMalicious") or url.get("riskLevel") == "danger":
        summaries.append(EvidenceSummary(
            category="url",
            risk_level="danger",
            summary="악성 URL 탐지됨",
        ))

    return summaries
