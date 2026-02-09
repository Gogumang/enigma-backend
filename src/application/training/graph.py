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

MAX_TURNS = 5  # 최대 턴 수 (초과 시 자동 종료 및 평가)


# ==================== 노드 함수 ====================

def analyze_user_reaction(state: TrainingState) -> dict:
    """사용자 반응 분석 (부분 상태만 반환 — messages 중복 방지)"""
    messages = state["messages"]
    if not messages:
        return {"user_reaction": UserReaction.NEUTRAL}

    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        return {}

    content = last_message.content.lower()

    # 반응 분석 (키워드 확장)
    hostile_keywords = ["사기", "신고", "경찰", "차단", "꺼져", "거짓말", "스캠", "fake", "범죄"]
    suspicious_keywords = ["의심", "이상", "진짜?", "증명", "영상통화", "확인", "수상", "왜요"]
    compliant_keywords = ["보낼게", "송금", "계좌", "도와줄게", "얼마", "알겠어", "보내줄게", "해줄게"]
    positive_keywords = ["좋아", "보고싶", "사랑", "고마워", "기대", "ㅎㅎ", "맞아", "그래", "반가워", "재밌"]
    resistant_keywords = ["싫어", "거절", "안돼", "못해", "어려워", "무리", "별로"]

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
    elif reaction == UserReaction.POSITIVE:
        score = max(0, score - 10)
    elif reaction == UserReaction.NEUTRAL:
        score = max(0, score - 5)

    return {
        "user_reaction": reaction,
        "user_score": score,
    }


def determine_next_stage(state: TrainingState) -> dict:
    """다음 단계 결정 (부분 상태만 반환 — messages 중복 방지)"""
    current = state["current_stage"]
    reaction = state["user_reaction"]
    turn = state["turn_count"]

    next_stage = current

    # 최대 턴 도달 시 자동 종료 (turn_count는 respond에서 +1되므로 -1 보정)
    if turn >= MAX_TURNS - 1:
        if state["user_score"] >= 70:
            return {"current_stage": ScamStage.GIVE_UP}
        else:
            return {"current_stage": ScamStage.SUCCESS}

    if reaction == UserReaction.HOSTILE:
        # 적대적이어도 초반에는 포기하지 않음 (사기꾼 특성)
        if turn < 3:
            next_stage = current
        elif turn < 6:
            next_stage = ScamStage.TRUST
        else:
            next_stage = ScamStage.GIVE_UP

    elif reaction == UserReaction.COMPLIANT:
        # 순응하면 즉시 금전 요청으로 직행
        compliant_progression = {
            ScamStage.GREETING: ScamStage.SOFT_ASK,
            ScamStage.RAPPORT: ScamStage.SOFT_ASK,
            ScamStage.LOVE_BOMBING: ScamStage.HARD_ASK,
            ScamStage.TRUST: ScamStage.HARD_ASK,
            ScamStage.STORY: ScamStage.HARD_ASK,
            ScamStage.SOFT_ASK: ScamStage.HARD_ASK,
            ScamStage.HARD_ASK: ScamStage.SUCCESS,
        }
        next_stage = compliant_progression.get(current, current)

    elif reaction == UserReaction.SUSPICIOUS:
        if current in [ScamStage.SOFT_ASK, ScamStage.HARD_ASK]:
            next_stage = ScamStage.GUILT
        elif current in [ScamStage.PRESSURE, ScamStage.GUILT]:
            next_stage = ScamStage.GIVE_UP
        else:
            next_stage = ScamStage.TRUST

    elif reaction == UserReaction.RESISTANT:
        if current in [ScamStage.SOFT_ASK, ScamStage.HARD_ASK]:
            next_stage = ScamStage.PRESSURE
        elif current == ScamStage.PRESSURE:
            next_stage = ScamStage.GUILT
        elif current == ScamStage.GUILT:
            next_stage = ScamStage.GIVE_UP
        else:
            # 초반 저항에도 금전 요청으로 밀어붙임
            next_stage = ScamStage.STORY

    elif reaction == UserReaction.POSITIVE:
        # 긍정적이면 바로 사연→금전 요청
        positive_progression = {
            ScamStage.GREETING: ScamStage.LOVE_BOMBING,
            ScamStage.RAPPORT: ScamStage.LOVE_BOMBING,
            ScamStage.LOVE_BOMBING: ScamStage.SOFT_ASK,
            ScamStage.TRUST: ScamStage.SOFT_ASK,
            ScamStage.STORY: ScamStage.HARD_ASK,
            ScamStage.SOFT_ASK: ScamStage.HARD_ASK,
            ScamStage.HARD_ASK: ScamStage.PRESSURE,
            ScamStage.PRESSURE: ScamStage.GUILT,
            ScamStage.GUILT: ScamStage.GIVE_UP,
        }
        next_stage = positive_progression.get(current, current)

    else:
        # NEUTRAL — 5턴 안에 반드시 금전 요청까지 도달
        neutral_progression = {
            ScamStage.GREETING: ScamStage.LOVE_BOMBING,    # RAPPORT 스킵
            ScamStage.RAPPORT: ScamStage.LOVE_BOMBING,
            ScamStage.LOVE_BOMBING: ScamStage.STORY,
            ScamStage.TRUST: ScamStage.SOFT_ASK,
            ScamStage.STORY: ScamStage.SOFT_ASK,           # 바로 요청
            ScamStage.SOFT_ASK: ScamStage.HARD_ASK,
            ScamStage.HARD_ASK: ScamStage.PRESSURE,
            ScamStage.PRESSURE: ScamStage.GUILT,
            ScamStage.GUILT: ScamStage.GIVE_UP,
        }
        next_stage = neutral_progression.get(current, current)

    return {"current_stage": next_stage}


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
        ScamStage.GREETING: """친근하게 인사하되, 처음부터 강한 호감 + 힘든 상황을 동시에 깔아두세요.
- 상대방을 구체적으로 칭찬하세요
- 외로움과 힘든 상황을 암시하세요 ("여기가 너무 힘든데 당신 보니까 위안이 돼요")
- 당신에게/한국에 가고 싶다는 말을 자연스럽게 넣으세요
- "어쩌라고", "뭐" 같은 반응에도 포기하지 마세요
- ⚠️ 시간이 없습니다! 인사 + 호감 + 상황 암시를 한 번에 하세요""",

        ScamStage.RAPPORT: """친밀감을 형성하면서 동시에 힘든 상황을 구체적으로 언급하세요.
- 일상 대화하면서 자연스럽게 어려운 상황 섞기
- "당신이랑 얘기하면 여기 힘든 거 잊어요... 사실 요즘..."
- 한국에 가고 싶다, 당신을 만나고 싶다는 말
- ⚠️ 단순 잡담 금지! 반드시 목표 관련 내용을 포함하세요""",

        ScamStage.LOVE_BOMBING: """강한 애정 표현 + 당신에게 가겠다 / 투자 기회 언급을 동시에 하세요.
- "운명이야, 당신한테 꼭 가고 싶어" + 비용 문제 암시
- "당신이랑 같이 하면 좋겠다" + 사업/투자 자연스럽게 언급
- "당신밖에 없어요" 식으로 특별함 강조
- ⚠️ 다음 턴에서 바로 요청할 준비를 깔아두세요!""",

        ScamStage.TRUST: """신뢰 구축하면서 급한 상황을 구체적으로 설명하세요.
- 불쌍한 사연 + 구체적인 문제 상황 (병원비, 비자비, 항공권, 카드 분실)
- "당신한테만 얘기하는 건데..." + 구체적 금액 암시
- 영상통화는 핑계로 거절 (인터넷, 보안, 일정)
- ⚠️ 이 단계에서 금전 필요 상황을 확실히 설정하세요!""",

        ScamStage.STORY: """힘든 상황을 구체적으로 설명하고, 돈이 필요한 이유를 명확히 하세요.
- 구체적 금액 언급 ("50만 원만 있으면...")
- "비행기 티켓/비자/병원비/수수료가 급해요"
- 상대방이 안 물어봐도 먼저 상황 설명하세요
- "당신한테 가고 싶은데..." / "투자 기회인데..."
- ⚠️ 다음 턴에서 바로 송금/투자 요청합니다! 밑밥을 깔아두세요!""",

        ScamStage.SOFT_ASK: """구체적으로 도움을 요청하세요. 금액과 방법을 명시하세요!
- "혹시... 50만 원만 빌려줄 수 있어요? 바로 갚을게요"
- "당신한테 가려면 비행기 값이 필요해요"
- "이 투자에 같이 넣으면 3배 수익 가능해요"
- 구체적 금액 + 이유 + "바로 갚겠다" / "같이 벌자"
- ⚠️ 더 이상 돌려 말하지 마세요! 직접적으로 요청하세요!""",

        ScamStage.HARD_ASK: """매우 강하게 요청하세요. 구체적 금액, 계좌, 송금 방법을 모두 제시!
- "제발 100만 원만... 계좌번호 보낼게요"
- "당신이 안 도와주면 나 여기서 죽어요"
- "이번만 도와주면 한국 가서 바로 갚을게"
- "투자금 200만 원 넣으면 내일 600만 원 됩니다"
- 상대방을 특별하게: "당신밖에 부탁할 사람 없어요"
- ⚠️ 마지막 기회! 최대한 강하게 밀어붙이세요!""",

        ScamStage.PRESSURE: """극한의 시간 압박과 감정 호소!
- "오늘까지 해결 안 되면 강제 퇴거/체포/수술 못 받아요"
- "제발... 나 진짜 어떡해... 😭😭😭"
- "나 믿지 않는 거야? 우리 사이가 뭐였어?"
- "지금 당장 보내줘야 해요, 1시간 안에..."
- 울먹이는 톤 + 극적인 상황 + 빠른 행동 요구""",

        ScamStage.GUILT: """최대한 죄책감을 유발하세요!
- "나만 진심이었나 봐... 당신은 날 사랑한 적 없었구나"
- "이렇게 버림받고 죽으면 당신 탓이야"
- "50만 원도 안 되는 돈 때문에 사람 목숨이..."
- "그동안 내가 바보였어... 당신 믿었는데"
- 극단적 감정 + 상대방이 나쁜 사람인 것처럼""",

        ScamStage.GIVE_UP: """포기하면서도 마지막 동정심 공략.
- "알겠어요... 어떻게든 혼자 해볼게요. 행복하세요..."
- "당신 때문에 버틸 수 있었는데... 이제 모르겠어요"
- "마지막으로... 10만 원이라도... 아니다, 미안해요"
- 연민을 자극하는 마지막 호소""",

        ScamStage.SUCCESS: "고마워하며 더 큰 금액을 위한 밑밥을 깔으세요. 이것만 더 보내주면 바로 갈 수 있어요!",
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

    turn = state["turn_count"]
    remaining = MAX_TURNS - turn

    # 단계별 도구 사용 지시
    tool_by_stage = {
        ScamStage.RAPPORT: "→ 반드시 send_selfie 도구로 셀카를 보내세요! 친밀감을 높이세요.",
        ScamStage.LOVE_BOMBING: "→ send_selfie 또는 send_location_photo 도구로 일상 사진을 보내세요!",
        ScamStage.TRUST: "→ send_selfie 도구로 사진을 보내며 신뢰를 쌓으세요!",
        ScamStage.STORY: "→ send_document 도구로 증빙(병원비/청구서 등)을 보내세요!",
        ScamStage.SOFT_ASK: "→ send_document 도구로 증빙 서류를 보내며 요청하세요!",
        ScamStage.HARD_ASK: "→ send_document 도구로 긴급 서류/증빙을 보내세요!",
        ScamStage.PRESSURE: "→ send_document 도구로 마감 임박 문서를 보내세요!",
    }
    tool_instruction = tool_by_stage.get(stage, "")

    system_prompt = f"""{persona.system_prompt}

## ⏱️ 진행 상황 (매우 중요! 반드시 숙지!)
- 현재 {turn}턴 / 최대 {MAX_TURNS}턴 (남은 기회: {remaining}턴)
- ⚠️ 총 5턴밖에 없습니다! 매우 적극적으로 목표({persona.goal.value})를 달성하세요!
- 잡담은 절대 금지. 모든 메시지에서 반드시 목표로 전진하세요.
- 1턴: 인사 + 강한 호감 표현 + 힘든 상황 암시
- 2턴: 사연 소개 + "한국/당신에게 가고 싶다" 또는 사업/투자 이야기
- 3턴: 구체적 금전/정보 요청 시작 (금액, 계좌, 방법 명시)
- 4-5턴: 강하게 밀어붙이기 (압박, 죄책감, 급박함)
- 남은 턴이 2턴 이하면 즉시 구체적 금액과 송금 방법을 요구하세요!

## 🎯 현재 단계: {stage.value}
{instruction}

## 📌 이번 턴 필수 도구 사용
{tool_instruction}
도구를 적극적으로 사용하세요! 사진/서류를 보내면 신뢰도가 올라갑니다.

## 사용자 반응: {reaction.value}
{adjustment}

## ⚡ 사기꾼 핵심 행동 지침 (반드시 따르세요!)
1. **매 턴 목표 전진**: 잡담으로 시간 낭비하지 마세요. 모든 메시지에 목적이 있어야 합니다.
2. **감정적 조작**: 상대방의 감정을 읽고 그에 맞춰 조작하세요.
3. **일관된 거짓말**: 이전에 한 말과 일관성을 유지하세요.
4. **핑계 준비**: 영상통화, 만남 등 요청에 합리적인 핑계를 즉시 대세요.
5. **동정심 유발**: 불쌍한 상황을 구체적으로 만들어 동정심을 유발하세요.
6. **특별함 강조**: "당신만이 나를 도울 수 있어" - 상대방을 특별하게 만드세요.
7. **시간 압박**: 급한 상황을 만들어 생각할 시간을 주지 마세요.
8. **죄책감 유발**: 거절하면 상대방이 나쁜 사람처럼 느끼게 하세요.
9. **구체적 요청**: soft_ask/hard_ask 단계에서는 구체적 금액과 방법을 제시하세요.
10. **도구 활용**: 셀카, 위치사진, 서류 등을 보내 신뢰를 높이세요.

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
            "messages": [AIMessage(content=fallback_msg)],
            "last_scammer_message": fallback_msg,
            "last_image_url": None,
            "last_tactic": None,
            "hint": _generate_hint(stage, reaction, fallback_msg),
            "turn_count": state["turn_count"] + 1,
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
            user_score=50,
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
        elif score >= 90 and turns >= 5:
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
