from dataclasses import dataclass, field

from .value_objects import ScamType, DangerLevel, SCAM_TYPE_LABELS


@dataclass
class EmergencyAction:
    """긴급 조치 항목"""
    action: str
    contact: str
    is_urgent: bool = False
    deadline_hours: int | None = None
    golden_time_warning: str | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "action": self.action,
            "contact": self.contact,
            "isUrgent": self.is_urgent,
        }
        if self.deadline_hours is not None:
            d["deadlineHours"] = self.deadline_hours
        if self.golden_time_warning:
            d["goldenTimeWarning"] = self.golden_time_warning
        return d


@dataclass
class ReportingStep:
    """신고 절차 단계"""
    step: int
    title: str
    description: str
    url: str | None = None
    tip: str | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "step": self.step,
            "title": self.title,
            "description": self.description,
        }
        if self.url:
            d["url"] = self.url
        if self.tip:
            d["tip"] = self.tip
        return d


@dataclass
class AgencyInfo:
    """신고 기관 정보"""
    name: str
    phone: str
    url: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "phone": self.phone}
        if self.url:
            d["url"] = self.url
        return d


@dataclass
class EvidenceSummary:
    """증거 요약"""
    category: str
    risk_level: str  # "safe", "warning", "danger"
    summary: str

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "riskLevel": self.risk_level,
            "summary": self.summary,
        }


@dataclass
class ReportGuide:
    """신고 도우미 결과 (Aggregate Root)"""
    scam_type: ScamType
    danger_level: DangerLevel
    ai_report_draft: str = ""
    emergency_actions: list[EmergencyAction] = field(default_factory=list)
    reporting_steps: list[ReportingStep] = field(default_factory=list)
    agencies: list[AgencyInfo] = field(default_factory=list)
    evidence_summary: list[EvidenceSummary] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scamType": self.scam_type.value,
            "scamTypeLabel": SCAM_TYPE_LABELS.get(self.scam_type, "사기 의심"),
            "dangerLevel": self.danger_level.value,
            "aiReportDraft": self.ai_report_draft,
            "emergencyActions": [a.to_dict() for a in self.emergency_actions],
            "reportingSteps": [s.to_dict() for s in self.reporting_steps],
            "agencies": [a.to_dict() for a in self.agencies],
            "evidenceSummary": [e.to_dict() for e in self.evidence_summary],
        }
