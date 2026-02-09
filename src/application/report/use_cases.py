"""
신고 도우미 유스케이스
- 도메인 서비스 조립
- OpenAI를 통한 AI 신고서 초안 생성
- 폴백 템플릿
"""
import logging
from datetime import datetime

from src.domain.report import ReportGuide
from src.domain.report.value_objects import ScamType, DangerLevel, SCAM_TYPE_LABELS
from src.domain.report.service import (
    classify_scam_type,
    determine_danger_level,
    build_emergency_actions,
    build_reporting_steps,
    build_agencies,
    build_evidence_summary,
)
from src.infrastructure.external.openai_service import OpenAIService

logger = logging.getLogger(__name__)


class GenerateReportGuideUseCase:
    """신고 도우미 생성 유스케이스"""

    def __init__(self, openai_service: OpenAIService) -> None:
        self._openai = openai_service

    async def execute(
        self,
        analysis_results: dict,
        damage_amount: int | None = None,
        damage_date: str | None = None,
        user_description: str | None = None,
    ) -> ReportGuide:
        # 1. 스캠 유형 분류 (도메인 서비스)
        scam_type = classify_scam_type(analysis_results)
        danger_level = determine_danger_level(analysis_results)

        # 2. 도메인 조립
        emergency_actions = build_emergency_actions(scam_type, danger_level)
        reporting_steps = build_reporting_steps(scam_type)
        agencies = build_agencies(scam_type)
        evidence_summary = build_evidence_summary(analysis_results)

        # 3. AI 신고서 초안 생성
        ai_report_draft = await self._generate_report_draft(
            scam_type=scam_type,
            danger_level=danger_level,
            evidence_summary=evidence_summary,
            analysis_results=analysis_results,
            damage_amount=damage_amount,
            damage_date=damage_date,
            user_description=user_description,
        )

        return ReportGuide(
            scam_type=scam_type,
            danger_level=danger_level,
            ai_report_draft=ai_report_draft,
            emergency_actions=emergency_actions,
            reporting_steps=reporting_steps,
            agencies=agencies,
            evidence_summary=evidence_summary,
        )

    async def _generate_report_draft(
        self,
        scam_type: ScamType,
        danger_level: DangerLevel,
        evidence_summary: list,
        analysis_results: dict,
        damage_amount: int | None,
        damage_date: str | None,
        user_description: str | None,
    ) -> str:
        """OpenAI로 신고서 초안 생성, 실패 시 폴백"""
        # 증거 텍스트
        evidence_text = "\n".join(
            f"- [{e.category}] {e.summary}" for e in evidence_summary
        ) or "- 분석 증거 없음"

        # 식별 정보 추출
        identifiers = self._extract_identifiers(analysis_results)

        prompt = f"""당신은 사이버범죄 피해 신고서 작성을 도와주는 전문가입니다.
아래 정보를 바탕으로 경찰청 사이버수사국에 제출할 **피해 신고서 초안**을 작성해주세요.

## 사기 유형: {SCAM_TYPE_LABELS.get(scam_type, '사기 의심')}
## 위험도: {danger_level.value}

## AI 분석 증거:
{evidence_text}

## 식별 정보:
{identifiers}

## 피해 정보:
- 피해 금액: {f'{damage_amount:,}원' if damage_amount else '[작성 필요]'}
- 피해 일시: {damage_date or '[작성 필요]'}
- 피해 설명: {user_description or '[작성 필요]'}

## 작성 지침:
1. 공식적이고 객관적인 어조로 작성
2. 피해자가 직접 채워야 하는 부분은 [작성 필요]로 표시
3. AI 분석 결과를 증거로 인용
4. 경찰 신고서 형식에 맞게 구조화
5. 한국어로 작성

아래 형식을 따라 작성하세요:

=== 사이버범죄 피해 신고서 ===

1. 신고인 정보
  - 성명: [작성 필요]
  - 연락처: [작성 필요]

2. 피해 유형
  - (위 분석 기반 작성)

3. 피의자 정보
  - (식별 정보 기반 작성)

4. 피해 내용
  - (분석 결과 + 피해 정보 기반 작성)

5. 증거 자료
  - (AI 분석 증거 기반 작성)

6. 요청 사항
  - (신고 목적 작성)
"""

        try:
            draft = await self._openai.generate_text(prompt)
            if draft and len(draft) > 50:
                return draft
        except Exception as e:
            logger.warning(f"AI 신고서 생성 실패, 폴백 사용: {e}")

        # 폴백 템플릿
        return self._fallback_report_draft(
            scam_type, evidence_text, identifiers,
            damage_amount, damage_date, user_description,
        )

    def _extract_identifiers(self, analysis_results: dict) -> str:
        """분석 결과에서 식별 정보 추출"""
        parts: list[str] = []

        fraud = analysis_results.get("fraud") or {}
        if fraud.get("value"):
            fraud_type = fraud.get("type", "")
            parts.append(f"- {fraud_type}: {fraud.get('displayValue', fraud.get('value', ''))}")

        profile = analysis_results.get("profile") or {}
        if profile.get("username"):
            parts.append(f"- SNS 계정: {profile.get('username', '')}")
        if profile.get("platform"):
            parts.append(f"- 플랫폼: {profile.get('platform', '')}")

        url = analysis_results.get("url") or {}
        if url.get("url"):
            parts.append(f"- URL: {url.get('url', '')}")

        return "\n".join(parts) if parts else "- 식별 정보 없음"

    def _fallback_report_draft(
        self,
        scam_type: ScamType,
        evidence_text: str,
        identifiers: str,
        damage_amount: int | None,
        damage_date: str | None,
        user_description: str | None,
    ) -> str:
        """AI 실패 시 템플릿 기반 폴백"""
        type_label = SCAM_TYPE_LABELS.get(scam_type, "사기 의심")
        today = datetime.now().strftime("%Y-%m-%d")

        return f"""=== 사이버범죄 피해 신고서 ===

작성일: {today}

1. 신고인 정보
  - 성명: [작성 필요]
  - 연락처: [작성 필요]
  - 주소: [작성 필요]

2. 피해 유형
  - 유형: {type_label}
  - 발생일시: {damage_date or '[작성 필요]'}

3. 피의자 정보 (AI 분석 기반)
{identifiers}

4. 피해 내용
  - 피해 금액: {f'{damage_amount:,}원' if damage_amount else '[작성 필요]'}
  - 피해 경위: {user_description or '[작성 필요 - 사건 경위를 시간순으로 상세히 기술해주세요]'}

5. 증거 자료 (AI 분석 결과)
{evidence_text}

6. 요청 사항
  - 피의자 수사 및 처벌
  - 피해 금액 환수 협조
  - 추가 피해 방지 조치

※ [작성 필요] 표시된 부분을 직접 작성해주세요.
※ 이 신고서는 AI가 자동 생성한 초안이며, 내용을 확인 후 제출해주세요."""
