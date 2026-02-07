from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel

from src.application.chat import AnalyzeChatUseCase
from src.interfaces.api.dependencies import (
    get_analyze_chat_use_case,
    get_openai_service,
)

router = APIRouter(prefix="/chat", tags=["chat"])


class AnalyzeRequest(BaseModel):
    messages: list[str]
    sender_id: str | None = None      # 메시지 발신자 ID (관계 분석용)
    receiver_id: str | None = None    # 메시지 수신자 ID (관계 분석용)


class ChatResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None


@router.post("/analyze", response_model=ChatResponse)
async def analyze_chat(
    request: AnalyzeRequest,
    use_case: AnalyzeChatUseCase = Depends(get_analyze_chat_use_case)
):
    """채팅 메시지 분석"""
    if not request.messages:
        return ChatResponse(
            success=False,
            error="분석할 메시지가 없습니다"
        )

    try:
        result = await use_case.execute(
            messages=request.messages,
            sender_id=request.sender_id,
            receiver_id=request.receiver_id
        )

        response_data = {
            "riskScore": result.risk_score,
            "riskCategory": result.risk_category,
            "detectedPatterns": result.detected_patterns,
            "warningSigns": result.warning_signs,
            "recommendations": result.recommendations,
            "aiAnalysis": result.ai_analysis,
            "interpretationSteps": result.interpretation_steps,
        }

        # RAG 컨텍스트 추가
        if result.rag_context:
            response_data["ragContext"] = result.rag_context

        # 파싱된 메시지 추가
        if result.parsed_messages:
            response_data["parsedMessages"] = result.parsed_messages

        # 관계 분석 컨텍스트 추가
        if result.relationship_context:
            response_data["relationshipContext"] = result.relationship_context

        # 원본 위험도 (관계 조정 전)
        if result.raw_risk_score is not None:
            response_data["rawRiskScore"] = result.raw_risk_score

        return ChatResponse(
            success=True,
            data=response_data
        )

    except Exception as e:
        return ChatResponse(success=False, error=str(e))


@router.post("/analyze-screenshot", response_model=ChatResponse)
async def analyze_screenshot(
    file: UploadFile = File(...),
):
    """채팅 스크린샷 분석"""
    # 파일 타입 검증
    if not file.content_type or not file.content_type.startswith("image/"):
        return ChatResponse(
            success=False,
            error="이미지 파일만 업로드 가능합니다"
        )

    try:
        # 이미지 데이터 읽기
        image_data = await file.read()

        # 파일 크기 제한 (10MB)
        if len(image_data) > 10 * 1024 * 1024:
            return ChatResponse(
                success=False,
                error="파일 크기는 10MB 이하여야 합니다"
            )

        # OpenAI 서비스로 분석
        openai_service = get_openai_service()

        # 스크린샷의 경우 사전 RAG 조회 없이 AI가 직접 분석
        rag_context = None

        result = await openai_service.analyze_chat_screenshot(image_data, rag_context)

        response_data = {
            "riskScore": result.risk_score,
            "riskCategory": result.risk_category.value,
            "detectedPatterns": result.detected_patterns,
            "warningSigns": result.warning_signs,
            "recommendations": result.recommendations,
            "aiAnalysis": result.ai_analysis,
            "interpretationSteps": result.interpretation_steps,
        }

        # 추출된 메시지 추가
        if result.parsed_messages:
            response_data["parsedMessages"] = [
                {"role": p.role.value, "content": p.content}
                for p in result.parsed_messages
            ]

        return ChatResponse(
            success=True,
            data=response_data
        )

    except Exception as e:
        return ChatResponse(
            success=False,
            error=f"스크린샷 분석 실패: {str(e)}"
        )
