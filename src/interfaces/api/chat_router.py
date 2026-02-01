from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel
from typing import Optional

from src.application.chat import AnalyzeChatUseCase, ChatbotUseCase, GetPatternsUseCase
from src.infrastructure.persistence import RelationshipType
from src.interfaces.api.dependencies import (
    get_analyze_chat_use_case,
    get_chatbot_use_case,
    get_patterns_use_case,
    get_openai_service,
    get_qdrant_repository,
    get_relationship_repository,
)

router = APIRouter(prefix="/chat", tags=["chat"])


class AnalyzeRequest(BaseModel):
    messages: list[str]
    sender_id: str | None = None      # 메시지 발신자 ID (관계 분석용)
    receiver_id: str | None = None    # 메시지 수신자 ID (관계 분석용)


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


# ==================== 관계 관리 API ====================

class SetRelationshipRequest(BaseModel):
    user_id: str
    other_user_id: str
    relationship_type: str  # friend, close_friend, family, lover, acquaintance, online_friend, matched, stranger
    trust_level: Optional[float] = None  # 0.0 ~ 1.0, None이면 기본값 사용
    platform: Optional[str] = None  # tinder, bumble, etc.


class GetRelationshipRequest(BaseModel):
    user_id: str
    other_user_id: str


@router.post("/relationship/set", response_model=ChatResponse)
async def set_relationship(request: SetRelationshipRequest):
    """사용자 간 관계 설정"""
    try:
        repo = get_relationship_repository()

        if not repo.is_connected():
            return ChatResponse(
                success=False,
                error="관계 DB가 연결되지 않았습니다"
            )

        # 관계 유형 변환
        try:
            rel_type = RelationshipType(request.relationship_type)
        except ValueError:
            valid_types = [t.value for t in RelationshipType]
            return ChatResponse(
                success=False,
                error=f"유효하지 않은 관계 유형입니다. 사용 가능: {valid_types}"
            )

        success = await repo.set_relationship(
            user_id=request.user_id,
            other_user_id=request.other_user_id,
            relationship_type=rel_type,
            trust_level=request.trust_level,
            platform=request.platform
        )

        if success:
            return ChatResponse(
                success=True,
                data={
                    "message": f"관계 설정 완료: {request.user_id} -> {request.other_user_id} ({rel_type.value})"
                }
            )
        else:
            return ChatResponse(success=False, error="관계 설정 실패")

    except Exception as e:
        return ChatResponse(success=False, error=str(e))


@router.post("/relationship/get", response_model=ChatResponse)
async def get_relationship(request: GetRelationshipRequest):
    """사용자 간 관계 조회"""
    try:
        repo = get_relationship_repository()

        if not repo.is_connected():
            return ChatResponse(
                success=False,
                error="관계 DB가 연결되지 않았습니다"
            )

        relationship = await repo.get_relationship(
            user_id=request.user_id,
            other_user_id=request.other_user_id
        )

        if relationship:
            return ChatResponse(
                success=True,
                data={
                    "relationship": {
                        "type": relationship.relationship_type.value,
                        "trustLevel": relationship.trust_level,
                        "knownSince": relationship.known_since.isoformat() if relationship.known_since else None,
                        "interactionCount": relationship.interaction_count,
                        "financialRequestCount": relationship.financial_request_count,
                        "platform": relationship.platform
                    }
                }
            )
        else:
            return ChatResponse(
                success=True,
                data={
                    "relationship": None,
                    "message": "등록된 관계가 없습니다"
                }
            )

    except Exception as e:
        return ChatResponse(success=False, error=str(e))


@router.get("/relationship/types")
async def get_relationship_types():
    """사용 가능한 관계 유형 목록"""
    return {
        "success": True,
        "data": {
            "types": [
                {"value": "friend", "label": "친구", "defaultTrust": 0.8},
                {"value": "close_friend", "label": "절친", "defaultTrust": 0.9},
                {"value": "family", "label": "가족", "defaultTrust": 0.95},
                {"value": "lover", "label": "연인", "defaultTrust": 0.85},
                {"value": "acquaintance", "label": "지인", "defaultTrust": 0.5},
                {"value": "online_friend", "label": "온라인 친구", "defaultTrust": 0.4},
                {"value": "matched", "label": "매칭앱에서 만남", "defaultTrust": 0.2},
                {"value": "stranger", "label": "모르는 사람", "defaultTrust": 0.1},
                {"value": "unknown", "label": "관계 불명", "defaultTrust": 0.3},
            ]
        }
    }
