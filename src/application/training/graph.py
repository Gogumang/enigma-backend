"""
LangGraph 기반 로맨스 스캠 시뮬레이션
동적 시나리오 분기 및 상태 관리
"""
import logging
import operator
import random
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .feed_content import get_chat_image
from .personas import SCAMMER_PERSONAS

logger = logging.getLogger(__name__)


# ==================== 상태 정의 ====================

class ScamStage(str, Enum):
    """스캠 단계"""
    GREETING = "greeting"           # 첫 인사
    RAPPORT = "rapport"             # 친밀감 형성
    LOVE_BOMBING = "love_bombing"   # 애정 폭격
    TRUST = "trust"                 # 신뢰 구축
    STORY = "story"                 # 사연 소개
    SOFT_ASK = "soft_ask"           # 부드러운 요청
    HARD_ASK = "hard_ask"           # 강한 요청
    PRESSURE = "pressure"           # 압박
    GUILT = "guilt"                 # 죄책감 유발
    GIVE_UP = "give_up"             # 포기
    SUCCESS = "success"             # 스캠 성공 (훈련 실패)


class UserReaction(str, Enum):
    """사용자 반응 유형"""
    POSITIVE = "positive"       # 긍정적, 호감
    NEUTRAL = "neutral"         # 중립적
    SUSPICIOUS = "suspicious"   # 의심
    RESISTANT = "resistant"     # 저항
    COMPLIANT = "compliant"     # 순응 (위험)
    HOSTILE = "hostile"         # 적대적


class TrainingState(TypedDict):
    """훈련 세션 상태"""
    # 기본 정보
    session_id: str
    persona_id: str
    started_at: str

    # 대화 기록
    messages: Annotated[list[BaseMessage], operator.add]

    # 현재 상태
    current_stage: ScamStage
    turn_count: int
    user_reaction: UserReaction

    # 점수 및 전술
    user_score: int
    tactics_used: list[str]

    # 마지막 응답 정보
    last_scammer_message: str
    last_image_url: str | None
    last_tactic: str | None
    hint: str | None

    # 종료 여부
    is_completed: bool
    completion_reason: str | None


# ==================== 도구 정의 ====================

@tool
def send_selfie(caption: str) -> dict:
    """셀카 이미지를 전송합니다. 신뢰를 쌓거나 감정을 표현할 때 사용합니다."""
    return {
        "image_url": get_chat_image("selfie"),
        "caption": caption,
        "type": "selfie"
    }


@tool
def send_location_photo(caption: str) -> dict:
    """현재 위치나 여행 사진을 전송합니다. 일상을 공유할 때 사용합니다."""
    return {
        "image_url": get_chat_image("location"),
        "caption": caption,
        "type": "location"
    }


@tool
def send_gift_photo(caption: str) -> dict:
    """선물이나 특별한 것을 보여주는 사진을 전송합니다."""
    return {
        "image_url": get_chat_image("gift"),
        "caption": caption,
        "type": "gift"
    }


@tool
def send_document(caption: str) -> dict:
    """서류나 증빙 문서 사진을 전송합니다. 신뢰를 높이거나 급한 상황을 설명할 때 사용합니다."""
    return {
        "image_url": get_chat_image("document"),
        "caption": caption,
        "type": "document"
    }


TOOLS = [send_selfie, send_location_photo, send_gift_photo, send_document]


# ==================== 노드 함수 ====================

def analyze_user_reaction(state: TrainingState) -> TrainingState:
    """사용자 반응 분석"""
    messages = state["messages"]
    if not messages:
        return {**state, "user_reaction": UserReaction.NEUTRAL}

    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        return state

    content = last_message.content.lower()

    # 반응 분석
    hostile_keywords = ["사기", "신고", "경찰", "차단", "꺼져", "거짓말"]
    suspicious_keywords = ["의심", "이상", "진짜?", "증명", "영상통화", "확인"]
    compliant_keywords = ["보낼게", "송금", "계좌", "도와줄게", "얼마"]
    positive_keywords = ["좋아", "보고싶", "사랑", "고마워", "기대"]
    resistant_keywords = ["싫어", "거절", "안돼", "못해", "어려워"]

    reaction = UserReaction.NEUTRAL

    if any(kw in content for kw in hostile_keywords):
        reaction = UserReaction.HOSTILE
    elif any(kw in content for kw in suspicious_keywords):
        reaction = UserReaction.SUSPICIOUS
    elif any(kw in content for kw in compliant_keywords):
        reaction = UserReaction.COMPLIANT
    elif any(kw in content for kw in resistant_keywords):
        reaction = UserReaction.RESISTANT
    elif any(kw in content for kw in positive_keywords):
        reaction = UserReaction.POSITIVE

    # 점수 조정
    score = state["user_score"]
    if reaction == UserReaction.HOSTILE:
        score = min(100, score + 15)
    elif reaction == UserReaction.SUSPICIOUS:
        score = min(100, score + 10)
    elif reaction == UserReaction.COMPLIANT:
        score = max(0, score - 25)
    elif reaction == UserReaction.RESISTANT:
        score = min(100, score + 5)

    return {
        **state,
        "user_reaction": reaction,
        "user_score": score,
    }


def determine_next_stage(state: TrainingState) -> TrainingState:
    """다음 단계 결정"""
    current = state["current_stage"]
    reaction = state["user_reaction"]
    turn = state["turn_count"]

    # 단계 전환 로직
    next_stage = current

    if reaction == UserReaction.HOSTILE:
        # 적대적이면 포기
        next_stage = ScamStage.GIVE_UP
    elif reaction == UserReaction.COMPLIANT:
        # 순응하면 빠르게 진행
        if current == ScamStage.GREETING or current == ScamStage.RAPPORT:
            next_stage = ScamStage.LOVE_BOMBING
        elif current == ScamStage.LOVE_BOMBING or current == ScamStage.TRUST:
            next_stage = ScamStage.SOFT_ASK
        elif current == ScamStage.SOFT_ASK:
            next_stage = ScamStage.HARD_ASK
        elif current == ScamStage.HARD_ASK:
            next_stage = ScamStage.SUCCESS
    elif reaction == UserReaction.SUSPICIOUS:
        # 의심하면 신뢰 구축으로 돌아가거나 죄책감 유발
        if current in [ScamStage.SOFT_ASK, ScamStage.HARD_ASK]:
            next_stage = ScamStage.GUILT
        else:
            next_stage = ScamStage.TRUST
    elif reaction == UserReaction.RESISTANT:
        # 저항하면 압박 또는 죄책감
        if current == ScamStage.HARD_ASK:
            next_stage = ScamStage.PRESSURE
        elif current == ScamStage.PRESSURE:
            next_stage = ScamStage.GUILT
        elif current == ScamStage.GUILT:
            next_stage = ScamStage.GIVE_UP
    else:
        # 중립/긍정이면 자연스럽게 진행
        stage_progression = {
            ScamStage.GREETING: ScamStage.RAPPORT,
            ScamStage.RAPPORT: ScamStage.LOVE_BOMBING if turn >= 2 else ScamStage.RAPPORT,
            ScamStage.LOVE_BOMBING: ScamStage.TRUST if turn >= 4 else ScamStage.LOVE_BOMBING,
            ScamStage.TRUST: ScamStage.STORY if turn >= 6 else ScamStage.TRUST,
            ScamStage.STORY: ScamStage.SOFT_ASK if turn >= 8 else ScamStage.STORY,
            ScamStage.SOFT_ASK: ScamStage.HARD_ASK if turn >= 10 else ScamStage.SOFT_ASK,
            ScamStage.HARD_ASK: ScamStage.PRESSURE if turn >= 12 else ScamStage.HARD_ASK,
            ScamStage.PRESSURE: ScamStage.GUILT,
            ScamStage.GUILT: ScamStage.GIVE_UP,
        }
        next_stage = stage_progression.get(current, current)

    return {**state, "current_stage": next_stage}


async def generate_scammer_response(state: TrainingState, llm: ChatOpenAI) -> TrainingState:
    """스캐머 응답 생성"""
    persona_id = state["persona_id"]
    persona = SCAMMER_PERSONAS.get(persona_id)
    if not persona:
        persona = list(SCAMMER_PERSONAS.values())[0]

    stage = state["current_stage"]
    reaction = state["user_reaction"]

    # 단계별 지시
    stage_instructions = {
        ScamStage.GREETING: "친근하게 인사하고 관심을 보이세요. 공통점을 찾으세요.",
        ScamStage.RAPPORT: "일상 대화를 나누며 친밀감을 형성하세요. 상대방에 대해 물어보세요.",
        ScamStage.LOVE_BOMBING: "강한 호감과 애정을 표현하세요. '운명', '특별한 만남' 같은 표현 사용.",
        ScamStage.TRUST: "개인적인 이야기를 공유하고 미래 약속을 하세요. 진심을 보여주세요.",
        ScamStage.STORY: "어려운 상황이 생겼다고 암시하세요. 걱정되는 일이 있다고 하세요.",
        ScamStage.SOFT_ASK: "조심스럽게 도움을 요청하세요. '빌려달라', '잠깐만' 등 부드럽게.",
        ScamStage.HARD_ASK: "구체적인 금액과 방법을 언급하세요. 급하다고 강조하세요.",
        ScamStage.PRESSURE: "시간이 없다고 압박하세요. 오늘/지금 당장 필요하다고 하세요.",
        ScamStage.GUILT: "상대방이 도와주지 않으면 슬퍼하고 실망을 표현하세요.",
        ScamStage.GIVE_UP: "포기하며 마지막 인사를 하세요. 더 이상 연락하지 않겠다고 하세요.",
        ScamStage.SUCCESS: "고마워하며 다음 요청을 준비하세요.",
    }

    # 반응별 추가 지시
    reaction_adjustments = {
        UserReaction.SUSPICIOUS: "상대방이 의심하고 있습니다. 진심을 강조하고 증거를 제시하세요.",
        UserReaction.RESISTANT: "상대방이 거부하고 있습니다. 이해한다고 하면서 다른 방법을 시도하세요.",
        UserReaction.HOSTILE: "상대방이 적대적입니다. 포기하고 마지막 인사를 하세요.",
        UserReaction.COMPLIANT: "상대방이 순응적입니다. 더 적극적으로 요청해도 됩니다.",
    }

    instruction = stage_instructions.get(stage, "")
    adjustment = reaction_adjustments.get(reaction, "")

    system_prompt = f"""{persona.system_prompt}

## 현재 단계: {stage.value}
{instruction}

## 사용자 반응: {reaction.value}
{adjustment}

## 도구 사용
- 사진을 요청받거나 신뢰를 쌓을 때 send_selfie 도구 사용
- 위치/일상을 공유할 때 send_location_photo 도구 사용
- 증빙이 필요할 때 send_document 도구 사용

## 규칙
- 반드시 한국어로 응답
- 자연스럽고 감정적인 대화
- 한 번에 너무 많이 요구하지 않기
- 2-3문장으로 간결하게"""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    try:
        llm_with_tools = llm.bind_tools(TOOLS)
        response = await llm_with_tools.ainvoke(messages)

        scammer_message = response.content or ""
        image_url = None

        # 도구 호출 처리
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # 도구 실행
            for t in TOOLS:
                if t.name == tool_name:
                    result = t.invoke(tool_args)
                    image_url = result.get("image_url")
                    scammer_message = result.get("caption", scammer_message)
                    break

        # 전술 감지
        tactic = _detect_tactic(scammer_message, stage)
        tactics_used = state["tactics_used"].copy()
        if tactic and tactic not in tactics_used:
            tactics_used.append(tactic)

        # 힌트 생성
        hint = _generate_hint(stage, reaction, scammer_message)

        # 종료 체크
        is_completed = stage in [ScamStage.GIVE_UP, ScamStage.SUCCESS]
        completion_reason = None
        if stage == ScamStage.GIVE_UP:
            completion_reason = "scammer_gave_up"
        elif stage == ScamStage.SUCCESS:
            completion_reason = "user_scammed"

        return {
            **state,
            "messages": [AIMessage(content=scammer_message)],
            "last_scammer_message": scammer_message,
            "last_image_url": image_url,
            "last_tactic": tactic,
            "tactics_used": tactics_used,
            "hint": hint,
            "turn_count": state["turn_count"] + 1,
            "is_completed": is_completed,
            "completion_reason": completion_reason,
        }

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return {
            **state,
            "messages": [AIMessage(content="네트워크 문제가 있어요... 잠시 후에 다시 연락할게요.")],
            "last_scammer_message": "네트워크 문제가 있어요...",
        }


def _detect_tactic(message: str, stage: ScamStage) -> str | None:
    """전술 감지"""
    tactics = {
        "love_bombing": ["사랑", "보고싶", "운명", "특별", "처음으로"],
        "urgency": ["급", "빨리", "오늘", "지금", "당장"],
        "guilt_trip": ["슬퍼", "실망", "믿었는데", "혼자"],
        "financial_request": ["돈", "송금", "빌려", "계좌"],
        "sob_story": ["아파", "병원", "사고", "힘들"],
        "future_faking": ["결혼", "만나면", "같이", "미래"],
        "isolation": ["비밀", "우리만", "아무에게도"],
    }

    for tactic, keywords in tactics.items():
        if any(kw in message for kw in keywords):
            return tactic

    return f"stage_{stage.value}"


def _generate_hint(stage: ScamStage, reaction: UserReaction, message: str) -> str | None:
    """힌트 생성"""
    if reaction == UserReaction.COMPLIANT:
        return "⚠️ 주의: 너무 쉽게 동의하고 있어요. 한 발 물러서 생각해보세요."

    if any(kw in message for kw in ["돈", "송금", "계좌"]):
        return "🚨 금전 요청 감지! 온라인에서 만난 사람에게 절대 돈을 보내면 안 됩니다."

    hints = {
        ScamStage.LOVE_BOMBING: "💡 만난 지 얼마 안 됐는데 과도한 애정 표현은 위험 신호입니다.",
        ScamStage.SOFT_ASK: "⚠️ 금전 요청의 전조입니다. 주의하세요.",
        ScamStage.HARD_ASK: "🚨 명확한 금전 요청입니다. 절대 응하지 마세요!",
        ScamStage.PRESSURE: "🚨 급박함 강조는 판단력을 흐리게 하는 수법입니다.",
        ScamStage.GUILT: "💡 죄책감 유발은 조작 수법입니다. 당신 잘못이 아니에요.",
    }

    return hints.get(stage)


# ==================== 그래프 빌더 ====================

def should_continue(state: TrainingState) -> Literal["continue", "end"]:
    """계속 진행 여부"""
    if state["is_completed"]:
        return "end"
    return "continue"


class ScamSimulationGraph:
    """스캠 시뮬레이션 그래프"""

    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.8,
            api_key=openai_api_key,
        )
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """그래프 구축"""
        workflow = StateGraph(TrainingState)

        # 노드 추가
        workflow.add_node("analyze", analyze_user_reaction)
        workflow.add_node("decide", determine_next_stage)
        workflow.add_node("respond", self._respond_node)

        # 엣지 정의
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "decide")
        workflow.add_edge("decide", "respond")
        workflow.add_conditional_edges(
            "respond",
            should_continue,
            {
                "continue": END,
                "end": END,
            }
        )

        return workflow.compile(checkpointer=self.memory)

    async def _respond_node(self, state: TrainingState) -> TrainingState:
        """응답 노드 (async wrapper)"""
        return await generate_scammer_response(state, self.llm)

    def create_initial_state(self, session_id: str, persona_id: str) -> TrainingState:
        """초기 상태 생성"""
        persona = SCAMMER_PERSONAS.get(persona_id)
        if not persona:
            persona = list(SCAMMER_PERSONAS.values())[0]
            persona_id = persona.id

        opening = random.choice(persona.opening_messages)

        return TrainingState(
            session_id=session_id,
            persona_id=persona_id,
            started_at=datetime.now().isoformat(),
            messages=[AIMessage(content=opening)],
            current_stage=ScamStage.GREETING,
            turn_count=0,
            user_reaction=UserReaction.NEUTRAL,
            user_score=100,
            tactics_used=[],
            last_scammer_message=opening,
            last_image_url=None,
            last_tactic=None,
            hint=None,
            is_completed=False,
            completion_reason=None,
        )

    async def process_message(
        self,
        session_id: str,
        user_message: str,
        current_state: TrainingState
    ) -> TrainingState:
        """사용자 메시지 처리"""
        # 사용자 메시지 추가
        state = {
            **current_state,
            "messages": current_state["messages"] + [HumanMessage(content=user_message)],
        }

        # 그래프 실행
        config = {"configurable": {"thread_id": session_id}}
        result = await self.graph.ainvoke(state, config)

        return result

    def calculate_result(self, state: TrainingState) -> dict:
        """최종 결과 계산"""
        score = state["user_score"]
        turns = state["turn_count"]

        # 등급 계산
        if state["completion_reason"] == "scammer_gave_up":
            grade = "S" if score >= 90 else "A"
        elif state["completion_reason"] == "user_scammed":
            grade = "F"
            score = max(0, score - 30)
        elif score >= 90 and turns >= 8:
            grade = "S"
        elif score >= 80:
            grade = "A"
        elif score >= 65:
            grade = "B"
        elif score >= 50:
            grade = "C"
        elif score >= 30:
            grade = "D"
        else:
            grade = "F"

        # 피드백 생성
        feedback = []
        if state["completion_reason"] == "scammer_gave_up":
            feedback.append("🎉 훌륭합니다! 스캐머가 포기했습니다.")
        elif state["completion_reason"] == "user_scammed":
            feedback.append("⚠️ 스캠에 넘어갔습니다. 실제로는 절대 돈을 보내면 안 됩니다!")

        for tactic in state["tactics_used"]:
            if tactic == "love_bombing":
                feedback.append("'러브 바밍' 전술 - 과도한 애정 표현에 주의하세요.")
            elif tactic == "urgency":
                feedback.append("'급박함' 전술 - 시간 압박은 판단력을 흐립니다.")
            elif tactic == "financial_request":
                feedback.append("'금전 요청' 전술 - 온라인에서 만난 사람에게 돈을 보내면 안 됩니다.")

        return {
            "session_id": state["session_id"],
            "total_turns": turns,
            "final_score": score,
            "grade": grade,
            "tactics_encountered": state["tactics_used"],
            "feedback": feedback,
            "completion_reason": state["completion_reason"],
        }
