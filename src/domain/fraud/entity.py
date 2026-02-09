from dataclasses import dataclass, field

from .value_objects import FraudStatus, FraudType, PhonePattern, AccountPattern


@dataclass
class FraudSource:
    """외부 API 조회 결과 하나"""
    source: str
    found: bool = False
    count: int = 0
    records: list = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "source": self.source,
            "found": self.found,
            "count": self.count,
        }
        if self.records:
            d["records"] = self.records
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class FraudCheckResult:
    """사기 조회 결과 (Aggregate Root)"""
    status: FraudStatus
    fraud_type: FraudType
    value: str
    display_value: str
    sources: list[FraudSource] = field(default_factory=list)
    total_records: int = 0
    message: str = ""
    recommendations: list[str] = field(default_factory=list)
    additional_links: list[dict] = field(default_factory=list)
    pattern_analysis: PhonePattern | AccountPattern | None = None
    bank: str | None = None

    def add_source(self, source: FraudSource) -> None:
        """소스 추가 및 상태 자동 갱신"""
        self.sources.append(source)
        if source.found:
            self.status = FraudStatus.DANGER
            self.total_records += source.count

    def build_message(self) -> None:
        """최종 메시지·권고사항 생성"""
        if self.status == FraudStatus.DANGER:
            self.message = f"⚠️ 사기 신고 이력이 발견되었습니다! ({self.total_records}건)"
            self.recommendations.extend([
                "이 번호/계좌와의 거래를 즉시 중단하세요",
                "이미 송금했다면 즉시 경찰(112)에 신고하세요",
                "금융감독원(1332)에 피해 상담을 받으세요",
            ])
        else:
            self.message = "현재까지 신고된 사기 이력이 없습니다."
            self.recommendations.extend([
                "신고 이력이 없다고 해서 100% 안전한 것은 아닙니다",
                "처음 거래하는 상대방에게는 소액 먼저 테스트하세요",
                "의심스러운 경우 경찰청 사이버안전국에 문의하세요",
            ])

        self.additional_links = [
            {
                "name": "경찰청 사이버범죄 신고",
                "url": "https://ecrm.police.go.kr",
                "description": "사이버범죄 신고 및 상담",
            },
            {
                "name": "금융감독원 보이스피싱 조회",
                "url": "https://www.fss.or.kr/fss/main/sub1sub3.do",
                "description": "피해계좌 조회 및 상담",
            },
        ]

    def to_dict(self) -> dict:
        d: dict = {
            "status": self.status.value,
            "type": self.fraud_type.value,
            "value": self.value,
            "displayValue": self.display_value,
            "sources": [s.to_dict() for s in self.sources],
            "totalRecords": self.total_records,
            "message": self.message,
            "recommendations": self.recommendations,
            "additionalLinks": self.additional_links,
        }
        if self.pattern_analysis:
            d["patternAnalysis"] = self.pattern_analysis.to_dict()
        if self.bank:
            d["bank"] = self.bank
        return d
