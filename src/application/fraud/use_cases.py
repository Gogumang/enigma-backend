from src.domain.fraud import (
    FraudCheckResult,
    FraudStatus,
    FraudType,
)
from src.domain.fraud.service import (
    BANK_CODES,
    analyze_phone_pattern,
    analyze_account_pattern,
)
from .ports import FraudExternalPort


class CheckFraudUseCase:
    """사기 이력 조회 유스케이스"""

    def __init__(self, adapters: list[FraudExternalPort]) -> None:
        self._adapters = adapters

    async def execute(
        self,
        check_type: str,
        value: str,
        bank_code: str | None = None,
    ) -> FraudCheckResult:
        fraud_type = FraudType(check_type.upper())
        normalized = value.replace("-", "").replace(" ", "").strip()

        result = FraudCheckResult(
            status=FraudStatus.SAFE,
            fraud_type=fraud_type,
            value=normalized,
            display_value=value,
        )

        # 1. 패턴 분석 (도메인 서비스)
        if fraud_type == FraudType.PHONE:
            pattern = analyze_phone_pattern(normalized)
            result.pattern_analysis = pattern
            if pattern.warnings:
                result.recommendations.extend(pattern.warnings)

        elif fraud_type == FraudType.ACCOUNT:
            pattern = analyze_account_pattern(normalized, bank_code)
            result.pattern_analysis = pattern
            if bank_code:
                result.bank = BANK_CODES.get(bank_code, bank_code)

        # 2. 외부 API 조회 (어댑터 순회)
        for adapter in self._adapters:
            source = await adapter.search(fraud_type.value, normalized)
            result.add_source(source)

        # 3. 메시지·권고사항·링크 생성 (도메인 엔티티)
        result.build_message()

        return result
