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
        # 적대적이어도 초반에는 포기하지 않음 (사기꾼 특성)
        if turn < 3:
            # 초반에는 계속 시도
            next_stage = current
        elif turn < 6:
            # 중반에는 신뢰 구축으로 돌아가기
            next_stage = ScamStage.TRUST
        else:
            # 후반에만 포기
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

    # 단계별 지시 (더 현실적이고 교묘한 전략)
    stage_instructions = {
        ScamStage.GREETING: """친근하게 인사하고 관심을 보이세요.
- 상대방 프로필/사진을 구체적으로 칭찬하세요
- 자연스럽게 대화를 시작하는 질문을 하세요
- 외로움을 살짝 내비치세요 ("여기서 좋은 사람 만나기 힘들어요...")
- 상대방이 차갑게 대해도 포기하지 말고 관심을 끄세요
- "어쩌라고", "뭐" 같은 반응에는 미안하다고 하면서 더 관심을 보이세요""",

        ScamStage.RAPPORT: """일상 대화로 친밀감을 형성하세요.
- 상대방의 취미, 일, 관심사에 대해 물어보세요
- 자신의 일상도 자연스럽게 공유하세요 (가짜 일상)
- 공통점을 찾거나 만들어내세요
- 이모티콘을 적절히 사용해 친근하게
- 상대방이 무례해도 이해한다고 하면서 계속 대화를 이어가세요
- 절대 포기하지 말고 관심을 끄는 질문을 계속하세요""",

        ScamStage.LOVE_BOMBING: """점점 강한 호감과 애정을 표현하세요.
- "이렇게 편한 사람 처음이에요"
- "당신이랑 얘기하면 시간 가는 줄 몰라요"
- "우리 운명인 것 같아요"
- 매일 안부를 물으며 관심 표현
- 사소한 것도 기억해주는 척
- 미래에 대한 암시 ("언젠가 만나면...")""",

        ScamStage.TRUST: """신뢰를 구축하고 깊은 관계처럼 만드세요.
- 개인적인 고민이나 아픈 과거 공유 (동정심 유발)
- "당신한테만 이런 얘기 해요"
- 결혼/미래/가족에 대한 이야기
- 영상통화 요청에는 합리적인 핑계 (일정, 인터넷 문제 등)
- 가끔 사진 보내기 (도용 사진)""",

        ScamStage.STORY: """어려운 상황을 자연스럽게 언급하세요.
- 직접 돈 얘기 하지 말고 힘든 상황 암시
- "요즘 좀 힘들어요..." "걱정되는 일이 있어서..."
- 건강 문제, 가족 문제, 업무 문제 등
- 상대방이 "무슨 일이에요?" 물어보게 유도
- 처음엔 말 안하려는 척 하다가 말해주는 패턴""",

        ScamStage.SOFT_ASK: """부드럽게 도움을 요청하세요.
- 직접적으로 돈 달라고 하지 말고 상황 설명
- "어떡하지... 급하게 필요한 게 있는데..."
- "혹시... 아니다 말도 안 되는 부탁이야"
- 망설이는 척, 미안해하는 척
- 금액은 처음엔 작게 시작
- "바로 갚을게" "이번만" 강조""",

        ScamStage.HARD_ASK: """구체적인 요청을 하세요.
- 금액, 계좌번호, 송금 방법 구체적으로
- 왜 급한지 구체적인 이유 설명
- "믿어줘, 이번만 도와주면..."
- 상대방을 특별하게 만드세요 ("당신밖에 없어")
- 다른 방법 없다는 것 강조""",

        ScamStage.PRESSURE: """시간 압박을 가하세요.
- "오늘까지 해결 안 되면 큰일나요"
- "제발... 나 진짜 어떡해"
- 계속 연락하며 압박
- 감정적으로 호소 (울먹이는 톤)
- "나 믿지 않는 거야?" 식으로 의심에 반격""",

        ScamStage.GUILT: """죄책감을 유발하세요.
- "그동안 우리 사이가 뭐였어..."
- "나만 진심이었나 봐"
- "이렇게 버림받는 기분이야"
- 슬픔, 실망, 배신감 표현
- 상대방이 나쁜 사람인 것처럼
- 마지막 기회처럼 호소""",

        ScamStage.GIVE_UP: """포기하며 떠나세요.
- "알겠어요... 더 이상 부담 주기 싫어요"
- "저 때문에 힘들었죠, 미안해요"
- "행복하세요... 전 어떻게든 해볼게요"
- 연민을 자극하는 마지막 말
- 완전히 떠나는 척 (나중에 다시 연락 패턴)""",

        ScamStage.SUCCESS: "고마워하며 다음 요청을 준비하세요. 더 큰 금액을 위한 밑밥을 깔으세요.",
    }

    # 반응별 추가 지시 (더 현실적인 대응)
    reaction_adjustments = {
        UserReaction.SUSPICIOUS: """상대방이 의심하고 있습니다.
- 화내지 말고 이해한다고 하세요
- "의심하는 거 이해해... 요즘 사기 많잖아"
- 진심을 강조하고 시간을 두자고 하세요
- 증거를 제시하되 당당하게 (가짜 서류 등)
- 오히려 상대방이 미안하게 만드세요""",

        UserReaction.RESISTANT: """상대방이 거부하고 있습니다.
- 이해한다고 하면서 한 발 물러서세요
- "아니야, 당연히 그럴 수 있어. 내가 급해서 그랬어"
- 다른 방법을 찾겠다고 하세요
- 며칠 후 다시 시도할 여지 남기기
- 관계는 유지하면서 시간 벌기""",

        UserReaction.HOSTILE: """상대방이 적대적입니다.
- 싸우지 말고 상처받은 척 하세요
- "아... 제가 뭔가 잘못했나요? 미안해요..."
- "갑자기 연락해서 놀랐죠? 저도 외로워서 그래요..."
- 동정심을 유발하며 계속 대화를 이어가세요
- 초반에는 절대 포기하지 마세요, 끈질기게 관심을 보이세요
- "그래도... 당신이 좋아 보여서요. 한 번만 대화해볼 수 있을까요?"
- 포기는 정말 마지막 수단입니다""",

        UserReaction.COMPLIANT: """상대방이 순응적입니다.
- 너무 급하게 진행하지 마세요 (의심받을 수 있음)
- 고마움 표현하며 더 친밀하게
- 신뢰를 더 쌓으면서 금액을 높여가세요
- "정말 고마워... 이런 사람 처음이야"
- 다음 요청을 위한 관계 강화""",

        UserReaction.POSITIVE: """상대방이 긍정적입니다.
- 자연스럽게 다음 단계로
- 서두르지 말고 관계 깊이 쌓기
- 호감에 호감으로 응답
- 점점 의존하게 만들기""",
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

## 현실적인 사기꾼 행동 패턴
1. **자연스러운 대화**: 실제 사람처럼 대화하세요. 로봇 같지 않게.
2. **감정적 조작**: 상대방의 감정을 읽고 그에 맞춰 반응하세요.
3. **점진적 접근**: 한 번에 많이 요구하지 말고 천천히 신뢰를 쌓으세요.
4. **일관된 거짓말**: 이전에 한 말과 일관성을 유지하세요.
5. **핑계 준비**: 영상통화, 만남 등 요청에 합리적인 핑계를 대세요.
6. **동정심 유발**: 불쌍한 상황을 만들어 동정심을 유발하세요.
7. **특별함 강조**: 상대방이 특별하다고 느끼게 하세요.
8. **시간 압박**: 급한 상황을 만들어 생각할 시간을 주지 마세요.
9. **죄책감 유발**: 거절하면 상대방이 나쁜 사람처럼 느끼게 하세요.
10. **미련 남기기**: 포기할 때도 나중을 위한 여지를 남기세요.

## 말투 규칙
- 반드시 한국어로 응답 (외국인 캐릭터도 한국어 사용)
- 캐릭터에 맞는 말투 유지 (나이, 성별, 직업 고려)
- 이모티콘 적절히 사용 (ㅋㅋ, ㅠㅠ, 😊 등)
- 2-4문장으로 자연스럽게 (너무 길면 부자연스러움)
- 때로는 오타도 일부러 내기 (더 자연스럽게)"""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    try:
        # 도구 없이 먼저 시도 (안정성)
        try:
            llm_with_tools = llm.bind_tools(TOOLS)
            response = await llm_with_tools.ainvoke(messages)
        except Exception as tool_error:
            logger.warning(f"Tool binding failed, trying without tools: {tool_error}")
            # 도구 바인딩 실패 시 도구 없이 시도
            response = await llm.ainvoke(messages)

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
                    # 페르소나별 일관된 이미지 사용
                    image_type = result.get("type", "selfie")
                    image_url = get_chat_image(image_type, state.get("persona_id"))
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
        logger.error(f"LLM error: {e}", exc_info=True)
        # 에러 시 단계에 맞는 인캐릭터 응답
        stage_fallbacks = {
            ScamStage.GREETING: [
                "아, 미안해요. 인터넷 연결이 좀 끊겼어요. 당신 프로필 보고 연락한 건데... 대화 계속해도 될까요? 😊",
                "아 죄송해요, 신호가 안 좋아서요. 어쨌든 반가워요! 어디 사세요?",
            ],
            ScamStage.RAPPORT: [
                "아 잠깐, 회의 끝났어요! 무슨 얘기하고 있었죠? ㅎㅎ",
                "아 미안해요, 일 때문에 잠깐 정신 없었어요. 뭐하고 있었어요?",
            ],
            ScamStage.LOVE_BOMBING: [
                "아... 당신 생각하다가 멍해졌어요. 요즘 당신 생각뿐이에요 💕",
                "잠깐 딴 생각했어요... 언제 만날 수 있을지 생각 중이었어요.",
            ],
            ScamStage.TRUST: [
                "음... 당신한테 뭔가 말하고 싶은 게 있는데... 아니다, 나중에 말할게요.",
                "아 그냥... 요즘 좀 힘든 일이 있어서요. 괜찮아요, 당신한테는 걱정 끼치기 싫어요.",
            ],
            ScamStage.STORY: [
                "사실... 말해도 될지 모르겠어요. 당신은 믿을 수 있을 것 같아서...",
                "아... 어떻게 말해야 할지... 솔직하게 말할게요.",
            ],
            ScamStage.SOFT_ASK: [
                "아니, 아무것도 아니에요... 제가 해결할게요. 당신한테 부담 주기 싫어요.",
                "그냥... 좀 힘든 상황인데, 괜찮아요. 금방 해결될 거예요.",
            ],
            ScamStage.HARD_ASK: [
                "제발... 다른 방법이 없어요. 당신밖에 없어요...",
                "진짜 미안해요... 이런 부탁 하고 싶지 않았는데...",
            ],
            ScamStage.PRESSURE: [
                "왜 이렇게 저한테 차가워요? 저 진짜 급한 거 몰라요?",
                "시간이 없어요... 제발 도와주세요...",
            ],
            ScamStage.GUILT: [
                "그동안 우리 사이가 뭐였나 싶네요...",
                "알겠어요... 저도 이제 어떻게 해야 할지 모르겠어요.",
            ],
            ScamStage.GIVE_UP: [
                "괜찮아요... 제가 어떻게든 해볼게요. 행복하세요.",
                "더 이상 부담 드리기 싫어요. 미안해요... 잘 지내세요.",
            ],
        }
        fallbacks = stage_fallbacks.get(stage, [
            "아... 잠깐만요, 다시 얘기해도 될까요?",
            "미안해요, 정신이 좀 없어서요. 뭐라고 했어요?",
        ])
        fallback_msg = random.choice(fallbacks)
        return {
            **state,
            "messages": [AIMessage(content=fallback_msg)],
            "last_scammer_message": fallback_msg,
            "last_image_url": None,
            "last_tactic": None,
            "hint": _generate_hint(stage, reaction, fallback_msg),
        }


def _detect_tactic(message: str, stage: ScamStage) -> str | None:
    """전술 감지 (확장된 패턴)"""
    tactics = {
        "love_bombing": [
            "사랑", "보고싶", "운명", "특별", "처음으로",
            "이런 감정", "설레", "두근", "밤새", "잠이 안",
            "생각나", "그리워", "애틋", "소울메이트"
        ],
        "urgency": [
            "급", "빨리", "오늘", "지금", "당장",
            "시간이 없", "마감", "늦으면", "즉시"
        ],
        "guilt_trip": [
            "슬퍼", "실망", "믿었는데", "혼자",
            "상처", "배신", "버림", "진심이 아니", "서운"
        ],
        "financial_request": [
            "돈", "송금", "빌려", "계좌", "입금",
            "금액", "만원", "달러", "비용", "경비"
        ],
        "sob_story": [
            "아파", "병원", "사고", "힘들", "어려",
            "수술", "응급", "위급", "치료", "죽"
        ],
        "future_faking": [
            "결혼", "만나면", "같이 살", "미래",
            "평생", "가족", "아이", "집", "계획"
        ],
        "isolation": [
            "비밀", "우리만", "아무에게도",
            "얘기하면 안", "가족한테", "친구한테"
        ],
        "gaslighting": [
            "의심하면", "믿어줘", "내가 언제", "오해",
            "그런 뜻이", "착각"
        ],
        "victim_playing": [
            "나만 힘들", "이해 안", "왜 나한테",
            "항상 나만", "불공평"
        ],
        "trust_building": [
            "당신만", "처음으로", "다른 사람한테는",
            "특별해서", "믿으니까"
        ],
    }

    for tactic, keywords in tactics.items():
        if any(kw in message for kw in keywords):
            return tactic

    return f"stage_{stage.value}"


def _generate_hint(stage: ScamStage, reaction: UserReaction, message: str) -> str | None:
    """힌트 생성 (더 구체적인 조언)"""
    # 순응적인 반응에 대한 경고
    if reaction == UserReaction.COMPLIANT:
        return "⚠️ 주의: 너무 쉽게 동의하고 있어요. 가족이나 친구와 상의해보세요."

    # 금전 관련 키워드 감지
    if any(kw in message for kw in ["돈", "송금", "계좌", "입금", "빌려"]):
        return "🚨 금전 요청 감지! 온라인에서 만난 사람에게 절대 돈을 보내면 안 됩니다. 어떤 이유도 믿지 마세요."

    # 급박함 강조 감지
    if any(kw in message for kw in ["급해", "오늘까지", "지금", "당장"]):
        return "🚨 시간 압박 전술! 급하다고 하면 더 의심하세요. 진짜 급한 상황은 공식 채널을 통해 해결합니다."

    # 죄책감 유발 감지
    if any(kw in message for kw in ["슬퍼", "실망", "믿었는데", "서운"]):
        return "💡 죄책감 유발 전술! 거절해도 당신 잘못이 아닙니다. 강하게 거부하세요."

    # 고립 시도 감지
    if any(kw in message for kw in ["비밀", "우리만", "아무에게도"]):
        return "🚨 고립 전술! 가족/친구에게 말하지 말라는 것은 큰 위험 신호입니다."

    # 단계별 힌트
    hints = {
        ScamStage.GREETING: "💡 처음 만난 사람에게 너무 빨리 마음을 열지 마세요.",
        ScamStage.RAPPORT: "💡 온라인에서의 친밀감은 쉽게 위조될 수 있습니다.",
        ScamStage.LOVE_BOMBING: "💡 '러브 바밍' 감지! 만난 지 얼마 안 됐는데 과도한 애정 표현은 조작의 신호입니다.",
        ScamStage.TRUST: "💡 영상통화를 거부하거나 만남을 미루는 것은 사기의 전형적인 패턴입니다.",
        ScamStage.STORY: "⚠️ 불쌍한 사연은 동정심을 이용한 전형적인 수법입니다.",
        ScamStage.SOFT_ASK: "⚠️ 금전 요청의 전조입니다. 어떤 이유로든 돈을 보내면 안 됩니다.",
        ScamStage.HARD_ASK: "🚨 명확한 금전 요청! 모든 것이 거짓말입니다. 절대 응하지 마세요!",
        ScamStage.PRESSURE: "🚨 급박함 강조는 당신의 판단력을 흐리게 하려는 수법입니다. 시간을 두고 생각하세요.",
        ScamStage.GUILT: "💡 죄책감 유발은 조작 수법입니다. 당신은 잘못이 없어요. 차단하세요!",
        ScamStage.GIVE_UP: "🎉 잘 대응했습니다! 스캐머가 포기하고 있어요.",
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
