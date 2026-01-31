from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel

from src.application.chat import AnalyzeChatUseCase, ChatbotUseCase, GetPatternsUseCase
from src.interfaces.api.dependencies import (
    get_analyze_chat_use_case,
    get_chatbot_use_case,
    get_patterns_use_case,
    get_openai_service,
    get_qdrant_repository,
)

router = APIRouter(prefix="/chat", tags=["chat"])


class AnalyzeRequest(BaseModel):
    messages: list[str]


class ChatRequest(BaseModel):
    message: str
    analyze: bool = False


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
        result = await use_case.execute(request.messages)

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

        return ChatResponse(
            success=True,
            data=response_data
        )

    except Exception as e:
        return ChatResponse(success=False, error=str(e))


@router.post("/message", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    use_case: ChatbotUseCase = Depends(get_chatbot_use_case)
):
    """챗봇에 메시지 전송"""
    if not request.message.strip():
        return ChatResponse(
            success=False,
            error="메시지를 입력해주세요"
        )

    try:
        result = await use_case.respond(request.message, request.analyze)

        data = {"message": result.message}

        if result.analysis:
            data["analysis"] = {
                "riskScore": result.analysis.risk_score,
                "riskCategory": result.analysis.risk_category,
                "detectedPatterns": result.analysis.detected_patterns,
                "warningSigns": result.analysis.warning_signs,
                "recommendations": result.analysis.recommendations,
                "aiAnalysis": result.analysis.ai_analysis
            }

        return ChatResponse(success=True, data=data)

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


@router.get("/patterns")
async def get_patterns(
    use_case: GetPatternsUseCase = Depends(get_patterns_use_case)
):
    """스캠 패턴 목록 조회"""
    patterns = use_case.execute()

    return {
        "success": True,
        "data": {
            "patterns": patterns
        }
    }
