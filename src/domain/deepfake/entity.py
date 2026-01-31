from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        if score >= 80:
            return cls.CRITICAL
        elif score >= 60:
            return cls.HIGH
        elif score >= 40:
            return cls.MEDIUM
        return cls.LOW


@dataclass
class DeepfakeAnalysis:
    """딥페이크 분석 결과 엔티티"""
    is_deepfake: bool
    confidence: float
    risk_level: RiskLevel
    media_type: MediaType
    details: dict
    analyzed_at: datetime

    @classmethod
    def create(
        cls,
        is_deepfake: bool,
        confidence: float,
        media_type: MediaType,
        details: dict | None = None
    ) -> "DeepfakeAnalysis":
        return cls(
            is_deepfake=is_deepfake,
            confidence=confidence,
            risk_level=RiskLevel.from_score(confidence if is_deepfake else 0),
            media_type=media_type,
            details=details or {},
            analyzed_at=datetime.now()
        )

    def to_dict(self) -> dict:
        return {
            "is_deepfake": self.is_deepfake,
            "confidence": self.confidence,
            "risk_level": self.risk_level.value,
            "media_type": self.media_type.value,
            "details": self.details,
            "analyzed_at": self.analyzed_at.isoformat()
        }
