from enum import Enum


class ScamType(str, Enum):
    """스캠 유형"""
    ROMANCE = "romance"
    VOICE_PHISHING = "voice_phishing"
    INVESTMENT = "investment"
    PHISHING = "phishing"
    UNKNOWN = "unknown"


SCAM_TYPE_LABELS: dict[ScamType, str] = {
    ScamType.ROMANCE: "로맨스 스캠",
    ScamType.VOICE_PHISHING: "보이스피싱",
    ScamType.INVESTMENT: "투자 사기",
    ScamType.PHISHING: "피싱",
    ScamType.UNKNOWN: "사기 의심",
}


class DangerLevel(str, Enum):
    """위험 수준"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
