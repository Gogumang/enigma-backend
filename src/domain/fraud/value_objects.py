from enum import Enum
from dataclasses import dataclass, field


class FraudStatus(str, Enum):
    SAFE = "safe"
    DANGER = "danger"


class FraudType(str, Enum):
    PHONE = "PHONE"
    ACCOUNT = "ACCOUNT"
    EMAIL = "EMAIL"


@dataclass
class PhonePattern:
    """전화번호 패턴 분석 결과"""
    is_valid: bool = False
    phone_type: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "isValid": self.is_valid,
            "type": self.phone_type,
            "warnings": self.warnings,
        }


@dataclass
class AccountPattern:
    """계좌번호 패턴 분석 결과"""
    is_valid: bool = False
    bank_code: str | None = None
    bank_name: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        result: dict = {
            "isValid": self.is_valid,
            "bank": self.bank_code,
            "warnings": self.warnings,
        }
        if self.bank_name:
            result["bankName"] = self.bank_name
        return result
