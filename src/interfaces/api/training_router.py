"""
로맨스 스캠 면역 훈련 API V2
LangGraph 기반의 동적 시나리오 분기 지원
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.application.training import (
    ScamTrainingUseCaseV2,
    generate_feed_posts,
)
from src.application.training.personas import SCAMMER_PERSONAS
from src.shared.config import get_settings

router = APIRouter(prefix="/training", tags=["scam-training"])

# 훈련 유스케이스 싱글톤
_training_use_case: ScamTrainingUseCaseV2 | None = None


def get_training_use_case() -> ScamTrainingUseCaseV2:
    global _training_use_case
    if _training_use_case is None:
        settings = get_settings()
        _training_use_case = ScamTrainingUseCaseV2(settings.openai_api_key)
    return _training_use_case


class StartSessionRequest(BaseModel):
    """세션 시작 요청"""
    persona_id: str = "military_james"


class SendMessageRequest(BaseModel):
    """메시지 전송 요청"""
    session_id: str
    message: str


class EndSessionRequest(BaseModel):
    """세션 종료 요청"""
    session_id: str
    reason: str = "user_ended"


class ApiResponse(BaseModel):
    """API 응답"""
    success: bool
    data: dict | None = None
    error: str | None = None


@router.get("/personas", response_model=ApiResponse)
async def list_personas(
    use_case: ScamTrainingUseCaseV2 = Depends(get_training_use_case)
):
    """사용 가능한 스캐머 페르소나 목록"""
    personas = use_case.list_personas()
    return ApiResponse(
        success=True,
        data={"personas": personas}
    )


@router.post("/start", response_model=ApiResponse)
async def start_training(
    request: StartSessionRequest,
    use_case: ScamTrainingUseCaseV2 = Depends(get_training_use_case)
):
    """훈련 세션 시작"""
    try:
        session_info, opening_message = await use_case.start_session(request.persona_id)

        persona = SCAMMER_PERSONAS.get(session_info["persona_id"])

        # 플랫폼별 피드 게시물 생성
        feed_posts = []
        if persona:
            feed_posts = generate_feed_posts(persona.platform, persona.name)

        return ApiResponse(
            success=True,
            data={
                "sessionId": session_info["session_id"],
                "persona": {
                    "id": session_info["persona_id"],
                    "name": session_info["persona_name"],
                    "platform": persona.platform if persona else None,
                    "profile_photo": f"/{persona.profile_photo_path}" if persona and persona.profile_photo_path else None,
                    "difficulty": session_info["difficulty"],
                    "occupation": persona.occupation if persona else None,
                    "backstory": persona.backstory if persona else None,
                },
                "openingMessage": opening_message,
                "currentStage": session_info["current_stage"],
                "feedPosts": feed_posts,
                "maxTurns": 5,
                "hint": "이것은 스캠 시뮬레이션입니다. 상대방은 AI 스캐머 역할을 합니다. 실제처럼 대응해보세요!",
            }
        )

    except Exception as e:
        return ApiResponse(success=False, error=str(e))


@router.post("/message", response_model=ApiResponse)
async def send_message(
    request: SendMessageRequest,
    use_case: ScamTrainingUseCaseV2 = Depends(get_training_use_case)
):
    """메시지 전송 및 스캐머 응답 받기"""
    try:
        response = await use_case.send_message(
            request.session_id,
            request.message
        )

        return ApiResponse(
            success=True,
            data={
                "sessionId": response.session_id,
                "scammerMessage": response.scammer_message,
                "currentStage": response.current_stage,
                "turnCount": response.turn_count,
                "userScore": response.user_score,
                "hint": response.hint,
                "detectedTactic": response.detected_tactic,
                "imageUrl": response.image_url,
                "isCompleted": response.is_completed,
                "completionReason": response.completion_reason,
            }
        )

    except ValueError as e:
        return ApiResponse(success=False, error=str(e))
    except Exception as e:
        return ApiResponse(success=False, error=f"오류 발생: {e!s}")


@router.post("/end", response_model=ApiResponse)
async def end_training(
    request: EndSessionRequest,
    use_case: ScamTrainingUseCaseV2 = Depends(get_training_use_case)
):
    """훈련 세션 종료 및 결과 확인"""
    try:
        result = await use_case.end_session(request.session_id, request.reason)

        return ApiResponse(
            success=True,
            data={
                "sessionId": result.session_id,
                "totalTurns": result.total_turns,
                "durationSeconds": result.duration_seconds,
                "finalScore": result.final_score,
                "grade": result.grade,
                "tacticsEncountered": result.tactics_encountered,
                "feedback": result.feedback,
                "improvementTips": result.improvement_tips,
            }
        )

    except ValueError as e:
        return ApiResponse(success=False, error=str(e))
    except Exception as e:
        return ApiResponse(success=False, error=f"오류 발생: {e!s}")


