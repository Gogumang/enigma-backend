"""
로맨스 스캠 면역 훈련 API
Fakebok 스타일의 스캠 시뮬레이션
"""
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.application.training import ScamTrainingUseCase
from src.interfaces.api.dependencies import get_openai_service

router = APIRouter(prefix="/training", tags=["scam-training"])

# 훈련 유스케이스 싱글톤
_training_use_case: Optional[ScamTrainingUseCase] = None


def get_training_use_case() -> ScamTrainingUseCase:
    global _training_use_case
    if _training_use_case is None:
        _training_use_case = ScamTrainingUseCase(get_openai_service())
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
    data: Optional[dict] = None
    error: Optional[str] = None


@router.get("/personas", response_model=ApiResponse)
async def list_personas(
    use_case: ScamTrainingUseCase = Depends(get_training_use_case)
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
    use_case: ScamTrainingUseCase = Depends(get_training_use_case)
):
    """훈련 세션 시작"""
    try:
        session, opening_message = await use_case.start_session(request.persona_id)

        return ApiResponse(
            success=True,
            data={
                "sessionId": session.id,
                "persona": {
                    "id": session.persona_id,
                    "name": session.persona_name,
                    "difficulty": session.difficulty,
                },
                "openingMessage": opening_message,
                "hint": "💡 이것은 스캠 시뮬레이션입니다. 상대방은 AI 스캐머 역할을 합니다. 실제처럼 대응해보세요!",
            }
        )

    except Exception as e:
        return ApiResponse(success=False, error=str(e))


@router.post("/message", response_model=ApiResponse)
async def send_message(
    request: SendMessageRequest,
    use_case: ScamTrainingUseCase = Depends(get_training_use_case)
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
                "currentPhase": response.current_phase,
                "turnCount": response.turn_count,
                "hint": response.hint,
                "detectedTactic": response.detected_tactic,
            }
        )

    except ValueError as e:
        return ApiResponse(success=False, error=str(e))
    except Exception as e:
        return ApiResponse(success=False, error=f"오류 발생: {str(e)}")


@router.post("/end", response_model=ApiResponse)
async def end_training(
    request: EndSessionRequest,
    use_case: ScamTrainingUseCase = Depends(get_training_use_case)
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
        return ApiResponse(success=False, error=f"오류 발생: {str(e)}")


@router.get("/session/{session_id}", response_model=ApiResponse)
async def get_session(
    session_id: str,
    use_case: ScamTrainingUseCase = Depends(get_training_use_case)
):
    """세션 상태 조회"""
    session = use_case.get_session(session_id)
    if not session:
        return ApiResponse(success=False, error="세션을 찾을 수 없습니다")

    return ApiResponse(
        success=True,
        data={
            "sessionId": session.id,
            "persona": {
                "id": session.persona_id,
                "name": session.persona_name,
            },
            "currentPhase": session.current_phase.value,
            "userScore": session.user_score,
            "turnCount": len([m for m in session.messages if m.role == "user"]),
            "isCompleted": session.is_completed,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in session.messages
            ],
        }
    )
