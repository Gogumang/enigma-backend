import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from src.domain.chat import ROMANCE_SCAM_PATTERNS, MessageParser, RAGContext, ChatAnalysisResult, ParsedMessage
from src.infrastructure.persistence import QdrantScamRepository
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
        qdrant_repo: Optional[QdrantScamRepository] = None
    ):
        self.ai_service = ai_service
        self.qdrant_repo = qdrant_repo

    async def execute(self, messages: list[str]) -> ChatAnalysisResponse:
        """채팅 메시지 분석"""
        if not messages:
            raise ValidationException("분석할 메시지가 없습니다")

        interpretation_steps = ["분석 시작"]

        # 1. 메시지 파싱 (발신자/수신자 구분)
        parsed_messages = MessageParser.parse_messages(messages)
        interpretation_steps.append(f"메시지 파싱 완료: {len(parsed_messages)}개 메시지")

        # 2. Qdrant 벡터 검색 (의미적 유사도)
        rag_context = None
        if self.qdrant_repo and self.qdrant_repo.is_connected():
            try:
                combined_text = " ".join(messages)
                rag_result = await self.qdrant_repo.search_similar(combined_text)

                rag_context = RAGContext(
                    matched_phrases=[
                        {
                            "text": p["text"],
                            "category": p["category"],
                            "severity": p["severity"],
                            "description": p["description"],
                            "similarity": p["similarity"],
                        }
                        for p in rag_result.matched_patterns
                    ],
                    similar_cases=[],  # Qdrant는 패턴만 저장
                    risk_indicators=rag_result.risk_indicators,
                    total_reports=0,
                )
                interpretation_steps.append(f"벡터 검색 완료: {len(rag_result.matched_patterns)}개 유사 패턴 발견")
            except Exception as e:
                logger.warning(f"Qdrant 검색 실패: {e}")
                interpretation_steps.append("벡터 검색 실패 - 기본 분석 수행")
        else:
            interpretation_steps.append("Qdrant 미연결 - 기본 분석 수행")

        # 3. AI 분석 (OpenAI 또는 Gemini)
        result = await self.ai_service.analyze_chat(
            messages=messages,
            rag_context=rag_context,
            parsed_messages=parsed_messages
        )

        return ChatAnalysisResponse(
            risk_score=result.risk_score,
            risk_category=result.risk_category.value,
            detected_patterns=result.detected_patterns,
            warning_signs=result.warning_signs,
            recommendations=result.recommendations,
            ai_analysis=result.ai_analysis,
            interpretation_steps=result.interpretation_steps or interpretation_steps,
            rag_context={
                "matched_phrases": rag_context.matched_phrases if rag_context else [],
                "similar_cases": rag_context.similar_cases if rag_context else [],
                "risk_indicators": rag_context.risk_indicators if rag_context else [],
                "total_reports": rag_context.total_reports if rag_context else 0,
            } if rag_context else None,
            parsed_messages=[
                {"role": p.role.value, "content": p.content}
                for p in parsed_messages
            ] if parsed_messages else None
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
