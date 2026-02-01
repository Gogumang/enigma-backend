"""
LangGraph 기반 로맨스 스캠 면역 훈련 유스케이스 V2
동적 시나리오 분기 및 상태 관리 지원
"""
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

from .graph import ScamSimulationGraph, TrainingState
from .personas import SCAMMER_PERSONAS

logger = logging.getLogger(__name__)


@dataclass
class TrainingResponseV2:
    """훈련 응답 V2"""
    session_id: str
    scammer_message: str
    current_stage: str
    turn_count: int
    user_score: int
    hint: str | None = None
    detected_tactic: str | None = None
    image_url: str | None = None
    is_completed: bool = False
    completion_reason: str | None = None


@dataclass
class TrainingResultV2:
    """훈련 결과 V2"""
    session_id: str
    total_turns: int
    duration_seconds: int
    final_score: int
    grade: str
    tactics_encountered: list[str]
    feedback: list[str]
    improvement_tips: list[str]


class ScamTrainingUseCaseV2:
    """LangGraph 기반 스캠 면역 훈련 V2"""

    def __init__(self, openai_api_key: str):
        self.graph = ScamSimulationGraph(openai_api_key)
        self.sessions: dict[str, TrainingState] = {}
        self.session_start_times: dict[str, datetime] = {}

    async def start_session(
        self,
        persona_id: str = "military_james"
    ) -> tuple[dict, str]:
        """훈련 세션 시작"""
        if persona_id not in SCAMMER_PERSONAS:
            persona_id = "military_james"

        session_id = str(uuid.uuid4())
        state = self.graph.create_initial_state(session_id, persona_id)

        self.sessions[session_id] = state
        self.session_start_times[session_id] = datetime.now()

        persona = SCAMMER_PERSONAS[persona_id]

        return {
            "session_id": session_id,
            "persona_id": persona_id,
            "persona_name": persona.name,
            "difficulty": persona.difficulty,
            "current_stage": state["current_stage"].value,
        }, state["last_scammer_message"]

    async def send_message(
        self,
        session_id: str,
        user_message: str
    ) -> TrainingResponseV2:
        """사용자 메시지 전송 및 응답 생성"""
        current_state = self.sessions.get(session_id)
        if not current_state:
            raise ValueError("세션을 찾을 수 없습니다")

        if current_state["is_completed"]:
            raise ValueError("이미 종료된 세션입니다")

        # LangGraph로 처리
        new_state = await self.graph.process_message(
            session_id, user_message, current_state
        )

        # 상태 업데이트
        self.sessions[session_id] = new_state

        return TrainingResponseV2(
            session_id=session_id,
            scammer_message=new_state["last_scammer_message"],
            current_stage=new_state["current_stage"].value,
            turn_count=new_state["turn_count"],
            user_score=new_state["user_score"],
            hint=new_state["hint"],
            detected_tactic=new_state["last_tactic"],
            image_url=new_state["last_image_url"],
            is_completed=new_state["is_completed"],
            completion_reason=new_state["completion_reason"],
        )

    async def end_session(
        self,
        session_id: str,
        reason: str = "user_ended"
    ) -> TrainingResultV2:
        """세션 종료 및 결과 분석"""
        state = self.sessions.get(session_id)
        if not state:
            raise ValueError("세션을 찾을 수 없습니다")

        # 종료 처리
        state = {**state, "is_completed": True, "completion_reason": reason}
        self.sessions[session_id] = state

        # 결과 계산
        result = self.graph.calculate_result(state)

        # 시간 계산
        start_time = self.session_start_times.get(session_id, datetime.now())
        duration = (datetime.now() - start_time).seconds

        # 개선 팁
        tips = self._generate_tips(state)

        return TrainingResultV2(
            session_id=session_id,
            total_turns=result["total_turns"],
            duration_seconds=duration,
            final_score=result["final_score"],
            grade=result["grade"],
            tactics_encountered=result["tactics_encountered"],
            feedback=result["feedback"],
            improvement_tips=tips,
        )

    def _generate_tips(self, state: TrainingState) -> list[str]:
        """개선 팁 생성"""
        tips = [
            "항상 영상통화를 요청하세요 - 거절하면 의심해야 합니다.",
            "가족이나 친구와 상황을 공유하세요.",
            "온라인에서 만난 사람에게 절대 돈을 보내지 마세요.",
            "너무 빠른 감정 표현은 경계하세요.",
            "상대방 정보를 독립적으로 확인해보세요.",
        ]

        if state["user_score"] < 70:
            tips.insert(0, "스캠 패턴에 대해 더 학습이 필요합니다.")

        if "financial_request" in state["tactics_used"]:
            tips.insert(0, "금전 요청에 더 단호하게 거절하세요.")

        return tips[:5]

    def get_session(self, session_id: str) -> dict | None:
        """세션 조회"""
        state = self.sessions.get(session_id)
        if not state:
            return None

        persona = SCAMMER_PERSONAS.get(state["persona_id"])

        return {
            "session_id": state["session_id"],
            "persona_id": state["persona_id"],
            "persona_name": persona.name if persona else "Unknown",
            "current_stage": state["current_stage"].value,
            "turn_count": state["turn_count"],
            "user_score": state["user_score"],
            "is_completed": state["is_completed"],
            "messages": [
                {
                    "role": "scammer" if hasattr(m, "type") and m.type == "ai" else "user",
                    "content": m.content,
                }
                for m in state["messages"]
            ],
        }

    def list_personas(self) -> list[dict]:
        """페르소나 목록"""
        return [
            {
                "id": p.id,
                "name": p.name,
                "occupation": p.occupation,
                "platform": p.platform,
                "profile_photo": f"/{p.profile_photo_path}" if p.profile_photo_path else None,
                "difficulty": p.difficulty,
                "goal": p.goal.value,
                "description": p.backstory[:100] + "...",
            }
            for p in SCAMMER_PERSONAS.values()
        ]
