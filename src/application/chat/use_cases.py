import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from src.domain.chat import ROMANCE_SCAM_PATTERNS, MessageParser, RAGContext, ChatAnalysisResult, ParsedMessage
from src.infrastructure.persistence import QdrantScamRepository, Neo4jRelationshipRepository
from src.shared.exceptions import ValidationException

logger = logging.getLogger(__name__)


@runtime_checkable
class AIService(Protocol):
    """AI 서비스 인터페이스"""
    async def analyze_chat(
        self,
        messages: list[str],
        rag_context: Optional[RAGContext] = None,
        parsed_messages: Optional[list[ParsedMessage]] = None
    ) -> ChatAnalysisResult: ...

    async def generate_response(self, user_message: str, context: str = "") -> str: ...


@dataclass
class ChatAnalysisResponse:
    """채팅 분석 응답 DTO"""
    risk_score: int
    risk_category: str
    detected_patterns: list[str]
    warning_signs: list[str]
    recommendations: list[str]
    ai_analysis: str
    # 추가 필드
    interpretation_steps: list[str] = field(default_factory=list)
    rag_context: dict | None = None
    parsed_messages: list[dict] | None = None
    # 관계 분석 결과
    relationship_context: dict | None = None
    raw_risk_score: int | None = None  # 관계 조정 전 원본 점수


@dataclass
class ChatResponse:
    """채팅 응답 DTO"""
    message: str
    analysis: ChatAnalysisResponse | None = None


class AnalyzeChatUseCase:
    """채팅 분석 유스케이스"""

    def __init__(
        self,
        ai_service: AIService,
        qdrant_repo: Optional[QdrantScamRepository] = None,
        relationship_repo: Optional[Neo4jRelationshipRepository] = None
    ):
        self.ai_service = ai_service
        self.qdrant_repo = qdrant_repo
        self.relationship_repo = relationship_repo

    async def execute(
        self,
        messages: list[str],
        sender_id: Optional[str] = None,
        receiver_id: Optional[str] = None
    ) -> ChatAnalysisResponse:
        """채팅 메시지 분석 (Qdrant + Neo4j 조합)"""
        if not messages:
            raise ValidationException("분석할 메시지가 없습니다")

        interpretation_steps = ["분석 시작"]

        # 1. 메시지 파싱 (발신자/수신자 구분)
        parsed_messages = MessageParser.parse_messages(messages)
        interpretation_steps.append(f"메시지 파싱 완료: {len(parsed_messages)}개 메시지")

        combined_text = " ".join(messages)

        # 2. 병렬 조회: Qdrant (패턴 유사도) + Neo4j (관계/맥락)
        rag_context = None
        relationship_result = None

        tasks = []

        # Qdrant 벡터 검색 태스크
        if self.qdrant_repo and self.qdrant_repo.is_connected():
            tasks.append(("qdrant", self.qdrant_repo.search_similar(combined_text)))
        else:
            interpretation_steps.append("Qdrant 미연결 - 패턴 매칭 생략")

        # Neo4j 관계 분석 태스크
        if self.relationship_repo and self.relationship_repo.is_connected():
            tasks.append(("neo4j", self.relationship_repo.analyze_relationship_context(
                message=combined_text,
                sender_id=sender_id,
                receiver_id=receiver_id
            )))
        else:
            interpretation_steps.append("Neo4j 미연결 - 관계 분석 생략")

        # 병렬 실행
        if tasks:
            task_names = [t[0] for t in tasks]
            task_coros = [t[1] for t in tasks]
            results = await asyncio.gather(*task_coros, return_exceptions=True)

            for name, result in zip(task_names, results):
                if isinstance(result, Exception):
                    logger.warning(f"{name} 조회 실패: {result}")
                    interpretation_steps.append(f"{name} 조회 실패")
                elif name == "qdrant":
                    # 위험 패턴과 안전 패턴 모두 포함
                    matched_phrases = [
                        {
                            "text": p["text"],
                            "category": p["category"],
                            "severity": p["severity"],
                            "description": p["description"],
                            "similarity": p["similarity"],
                            "is_safe": p.get("is_safe", False),
                        }
                        for p in result.matched_patterns
                    ]

                    # 안전 패턴 추가
                    safe_phrases = [
                        {
                            "text": p["text"],
                            "category": p["category"],
                            "severity": p["severity"],
                            "description": p["description"],
                            "similarity": p["similarity"],
                            "is_safe": True,
                        }
                        for p in result.safe_patterns
                    ]

                    rag_context = RAGContext(
                        matched_phrases=matched_phrases + safe_phrases,
                        similar_cases=[],
                        risk_indicators=result.risk_indicators,
                        total_reports=0,
                    )

                    # 보호 지표도 기록
                    if result.protective_indicators:
                        for indicator in result.protective_indicators[:3]:
                            interpretation_steps.append(f"✓ {indicator}")

                    interpretation_steps.append(
                        f"벡터 검색 완료: {len(result.matched_patterns)}개 위험 패턴, "
                        f"{len(result.safe_patterns)}개 안전 패턴"
                    )
                elif name == "neo4j":
                    relationship_result = result
                    if result.relationship:
                        interpretation_steps.append(
                            f"관계 분석 완료: {result.relationship.relationship_type.value} "
                            f"(신뢰도: {result.relationship.trust_level:.0%})"
                        )
                    if result.context:
                        interpretation_steps.append(
                            f"맥락 감지: {result.context.context_type} "
                            f"(키워드: {', '.join(result.context.keywords)})"
                        )

        # 3. AI 분석 (OpenAI 또는 Gemini)
        # 관계/맥락 정보를 AI에 전달
        relationship_context_for_ai = None
        if relationship_result:
            relationship_context_for_ai = {
                "relationship": {
                    "type": relationship_result.relationship.relationship_type.value,
                    "trust_level": relationship_result.relationship.trust_level,
                    "interaction_count": relationship_result.relationship.interaction_count,
                    "financial_request_count": relationship_result.relationship.financial_request_count,
                } if relationship_result.relationship else None,
                "context": {
                    "type": relationship_result.context.context_type,
                    "keywords": relationship_result.context.keywords,
                    "confidence": relationship_result.context.confidence,
                } if relationship_result.context else None,
                "trust_modifier": relationship_result.trust_modifier,
                "risk_factors": relationship_result.risk_factors,
                "protective_factors": relationship_result.protective_factors,
            }

        result = await self.ai_service.analyze_chat(
            messages=messages,
            rag_context=rag_context,
            parsed_messages=parsed_messages,
            relationship_context=relationship_context_for_ai
        )

        # 4. 관계 기반 위험도 조정
        raw_risk_score = result.risk_score
        adjusted_risk_score = result.risk_score

        if relationship_result:
            trust_modifier = relationship_result.trust_modifier
            # 신뢰도가 높으면 위험도 낮춤: final = raw * (1 - trust * 0.7)
            # 예: trust=0.9 (친구) → final = raw * 0.37
            # 예: trust=0.1 (모르는 사람) → final = raw * 0.93
            adjustment_factor = 1 - (trust_modifier * 0.7)
            adjusted_risk_score = int(raw_risk_score * adjustment_factor)

            interpretation_steps.append(
                f"위험도 조정: {raw_risk_score} → {adjusted_risk_score} "
                f"(신뢰 수정자: {trust_modifier:.2f})"
            )

            # 보호 요소/위험 요소 추가
            for factor in relationship_result.protective_factors:
                interpretation_steps.append(f"✓ 보호 요소: {factor}")
            for factor in relationship_result.risk_factors:
                interpretation_steps.append(f"⚠ 위험 요소: {factor}")

        # 관계 컨텍스트 딕셔너리 생성
        relationship_context_dict = None
        if relationship_result:
            relationship_context_dict = {
                "relationship": {
                    "type": relationship_result.relationship.relationship_type.value,
                    "trust_level": relationship_result.relationship.trust_level,
                    "interaction_count": relationship_result.relationship.interaction_count,
                    "financial_request_count": relationship_result.relationship.financial_request_count,
                } if relationship_result.relationship else None,
                "context": {
                    "type": relationship_result.context.context_type,
                    "keywords": relationship_result.context.keywords,
                    "confidence": relationship_result.context.confidence,
                } if relationship_result.context else None,
                "trust_modifier": relationship_result.trust_modifier,
                "risk_factors": relationship_result.risk_factors,
                "protective_factors": relationship_result.protective_factors,
            }

        return ChatAnalysisResponse(
            risk_score=adjusted_risk_score,
            risk_category=result.risk_category.value,
            detected_patterns=result.detected_patterns,
            warning_signs=result.warning_signs,
            recommendations=result.recommendations,
            ai_analysis=result.ai_analysis,
            interpretation_steps=interpretation_steps,
            rag_context={
                "matched_phrases": rag_context.matched_phrases if rag_context else [],
                "similar_cases": rag_context.similar_cases if rag_context else [],
                "risk_indicators": rag_context.risk_indicators if rag_context else [],
                "total_reports": rag_context.total_reports if rag_context else 0,
            } if rag_context else None,
            parsed_messages=[
                {"role": p.role.value, "content": p.content}
                for p in parsed_messages
            ] if parsed_messages else None,
            relationship_context=relationship_context_dict,
            raw_risk_score=raw_risk_score if relationship_result else None
        )


class ChatbotUseCase:
    """챗봇 대화 유스케이스"""

    def __init__(self, ai_service: AIService):
        self.ai_service = ai_service

    async def respond(
        self,
        user_message: str,
        analyze_for_scam: bool = False
    ) -> ChatResponse:
        """사용자 메시지에 응답"""
        if not user_message.strip():
            raise ValidationException("메시지를 입력해주세요")

        # AI 응답 생성
        response_text = await self.ai_service.generate_response(user_message)

        # 스캠 분석 요청 시
        analysis = None
        if analyze_for_scam:
            result = await self.ai_service.analyze_chat([user_message])
            analysis = ChatAnalysisResponse(
                risk_score=result.risk_score,
                risk_category=result.risk_category.value,
                detected_patterns=result.detected_patterns,
                warning_signs=result.warning_signs,
                recommendations=result.recommendations,
                ai_analysis=result.ai_analysis
            )

        return ChatResponse(
            message=response_text,
            analysis=analysis
        )


class GetPatternsUseCase:
    """스캠 패턴 조회 유스케이스"""

    def execute(self) -> list[dict]:
        """모든 스캠 패턴 반환"""
        return [
            {
                "type": p.pattern_type,
                "description": p.description,
                "severity": p.severity,
                "examples": p.examples
            }
            for p in ROMANCE_SCAM_PATTERNS
        ]
