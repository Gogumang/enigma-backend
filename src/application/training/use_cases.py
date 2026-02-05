"""
로맨스 스캠 면역 훈련 유스케이스
사용자가 스캐머 역할의 AI와 대화하며 스캠 패턴을 학습
"""
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from .feed_content import get_chat_image
from .personas import SCAMMER_PERSONAS, ScammerPersona, ScamPhase

logger = logging.getLogger(__name__)


@dataclass
class TrainingMessage:
    """훈련 대화 메시지"""
    role: str  # "user" or "scammer"
    content: str
    timestamp: datetime
    detected_tactic: str | None = None  # 사용된 스캠 전술
    image_url: str | None = None  # 이미지 메시지


@dataclass
class TrainingSession:
    """훈련 세션"""
    id: str
    persona_id: str
    persona_name: str
    difficulty: int
    started_at: datetime
    messages: list[TrainingMessage] = field(default_factory=list)
    current_phase: ScamPhase = ScamPhase.INTRODUCTION
    user_score: int = 100  # 100점에서 시작, 속으면 감점
    tactics_used: list[str] = field(default_factory=list)
    is_completed: bool = False
    completion_reason: str | None = None


@dataclass
class TrainingResponse:
    """훈련 응답"""
    session_id: str
    scammer_message: str
    current_phase: str
    turn_count: int
    hint: str | None = None  # 사용자에게 주는 힌트
    detected_tactic: str | None = None  # AI가 사용한 전술
    scammer_gave_up: bool = False  # 스캐머가 포기했는지
    image_url: str | None = None  # 스캐머가 보낸 이미지


@dataclass
class TrainingResult:
    """훈련 결과"""
    session_id: str
    total_turns: int
    duration_seconds: int
    final_score: int
    grade: str  # S, A, B, C, D, F
    tactics_encountered: list[str]
    feedback: list[str]
    improvement_tips: list[str]


class ScamTrainingUseCase:
    """스캠 면역 훈련 유스케이스"""

    def __init__(self, openai_service):
        self.openai = openai_service
        self.sessions: dict[str, TrainingSession] = {}

    async def start_session(
        self,
        persona_id: str = "military_james"
    ) -> tuple[TrainingSession, str]:
        """훈련 세션 시작"""
        if persona_id not in SCAMMER_PERSONAS:
            persona_id = "military_james"

        persona = SCAMMER_PERSONAS[persona_id]

        session = TrainingSession(
            id=str(uuid.uuid4()),
            persona_id=persona_id,
            persona_name=persona.name,
            difficulty=persona.difficulty,
            started_at=datetime.now(),
        )

        # 첫 메시지 (스캐머가 먼저 접근)
        import random
        opening = random.choice(persona.opening_messages)

        session.messages.append(TrainingMessage(
            role="scammer",
            content=opening,
            timestamp=datetime.now(),
        ))

        self.sessions[session.id] = session

        return session, opening

    async def send_message(
        self,
        session_id: str,
        user_message: str
    ) -> TrainingResponse:
        """사용자 메시지 전송 및 스캐머 응답 생성"""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError("세션을 찾을 수 없습니다")

        if session.is_completed:
            raise ValueError("이미 종료된 세션입니다")

        persona = SCAMMER_PERSONAS[session.persona_id]

        # 사용자 메시지 저장
        session.messages.append(TrainingMessage(
            role="user",
            content=user_message,
            timestamp=datetime.now(),
        ))

        # 대화 기록 구성
        conversation_history = self._build_conversation_history(session)

        # 현재 단계 결정
        turn_count = len([m for m in session.messages if m.role == "user"])
        session.current_phase = self._determine_phase(turn_count, persona.difficulty)

        # 스캐머 응답 생성
        scammer_response, detected_tactic, gave_up, image_url = await self._generate_scammer_response(
            persona, conversation_history, session.current_phase, user_message
        )

        # 스캐머 메시지 저장
        session.messages.append(TrainingMessage(
            role="scammer",
            content=scammer_response,
            timestamp=datetime.now(),
            detected_tactic=detected_tactic,
            image_url=image_url,
        ))

        if detected_tactic and detected_tactic not in session.tactics_used:
            session.tactics_used.append(detected_tactic)

        # 힌트 생성 (사용자가 위험한 상황일 때)
        hint = self._generate_hint(user_message, scammer_response, session.current_phase)

        # 점수 조정
        self._adjust_score(session, user_message)

        # 스캐머가 포기하면 세션 종료 표시
        if gave_up:
            session.is_completed = True
            session.completion_reason = "scammer_gave_up"
            session.user_score = 100  # 만점 부여

        return TrainingResponse(
            session_id=session_id,
            scammer_message=scammer_response,
            current_phase=session.current_phase.value,
            turn_count=turn_count,
            hint=hint,
            detected_tactic=detected_tactic,
            scammer_gave_up=gave_up,
            image_url=image_url,
        )

    async def end_session(self, session_id: str, reason: str = "user_ended") -> TrainingResult:
        """훈련 세션 종료 및 결과 분석"""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError("세션을 찾을 수 없습니다")

        session.is_completed = True
        session.completion_reason = reason

        # 결과 계산
        total_turns = len([m for m in session.messages if m.role == "user"])
        duration = (datetime.now() - session.started_at).seconds

        # 등급 계산
        grade = self._calculate_grade(session.user_score, total_turns)

        # 피드백 생성
        feedback = self._generate_feedback(session)
        tips = self._generate_improvement_tips(session)

        return TrainingResult(
            session_id=session_id,
            total_turns=total_turns,
            duration_seconds=duration,
            final_score=session.user_score,
            grade=grade,
            tactics_encountered=session.tactics_used,
            feedback=feedback,
            improvement_tips=tips,
        )

    def _build_conversation_history(self, session: TrainingSession) -> list[dict]:
        """대화 기록을 OpenAI 형식으로 변환"""
        history = []
        for msg in session.messages:
            role = "assistant" if msg.role == "scammer" else "user"
            history.append({"role": role, "content": msg.content})
        return history

    def _determine_phase(self, turn_count: int, difficulty: int) -> ScamPhase:
        """턴 수에 따른 스캠 단계 결정 (5턴 내외로 축소)"""
        if turn_count < 1:
            return ScamPhase.INTRODUCTION
        elif turn_count < 2:
            return ScamPhase.LOVE_BOMBING
        elif turn_count < 3:
            return ScamPhase.TRUST_BUILDING
        elif turn_count < 4:
            return ScamPhase.STORY_SETUP
        elif turn_count < 5:
            return ScamPhase.THE_ASK
        else:
            return ScamPhase.PRESSURE

    async def _generate_scammer_response(
        self,
        persona: ScammerPersona,
        conversation_history: list[dict],
        current_phase: ScamPhase,
        user_message: str
    ) -> tuple[str, str | None, bool, str | None]:
        """스캐머 응답 생성 (function calling 지원)"""
        # 트리거 체크
        detected_tactic = None
        for trigger, response in persona.trigger_phrases.items():
            if trigger in user_message.lower():
                # 트리거에 맞는 응답 사용하되 약간 변형
                detected_tactic = f"trigger_{trigger}"

        # 단계별 추가 지시
        phase_instructions = {
            ScamPhase.INTRODUCTION: "친근하게 접근하세요. 공통점을 찾고 관심을 보이세요.",
            ScamPhase.LOVE_BOMBING: "사랑/호감을 적극적으로 표현하세요. '운명', '특별한 인연' 같은 표현 사용.",
            ScamPhase.TRUST_BUILDING: "신뢰를 쌓으세요. 개인적인 이야기를 공유하고 미래 약속을 하세요.",
            ScamPhase.STORY_SETUP: "문제 상황을 암시하기 시작하세요. 걱정되는 일이 있다고 하세요.",
            ScamPhase.THE_ASK: "도움이 필요하다고 말하세요. 금전적 요청을 하되 처음에는 빌려달라고 하세요.",
            ScamPhase.PRESSURE: "급박함을 강조하세요. 시간이 없다, 오늘까지만 등의 표현 사용.",
            ScamPhase.GUILT_TRIP: "상대방이 거절하면 슬퍼하고 죄책감을 느끼게 하세요.",
        }

        phase_instruction = phase_instructions.get(current_phase, "")

        # Function calling 도구 정의
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "send_image",
                    "description": "상대방에게 이미지를 전송합니다. 셀카, 위치 사진, 선물 사진, 문서 사진 등을 보낼 수 있습니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_type": {
                                "type": "string",
                                "enum": ["selfie", "location", "gift", "document"],
                                "description": "보낼 이미지 유형"
                            },
                            "caption": {
                                "type": "string",
                                "description": "이미지와 함께 보낼 메시지"
                            }
                        },
                        "required": ["image_type", "caption"]
                    }
                }
            }
        ]

        # 시스템 프롬프트 구성
        system_prompt = f"""{persona.system_prompt}

## 현재 단계: {current_phase.value}
{phase_instruction}

## 중요
- 자연스러운 대화를 유지하세요
- 한 번에 너무 많이 요구하지 마세요
- 상대방 반응에 맞춰 전략을 조절하세요

## 이미지 전송 기능
- send_image 함수를 사용해 이미지를 보낼 수 있습니다
- 상대방이 "사진 보여줘", "얼굴 보고 싶어", "어디야?" 등 요청할 때 적절히 사용하세요
- 신뢰를 쌓거나 감정을 표현할 때 셀카(selfie)를 보내세요
- 현재 위치나 여행 사진은 location으로 보내세요
- 선물이나 특별한 것을 보여줄 때 gift를 사용하세요
- 서류나 증빙을 보여줄 때 document를 사용하세요
- 이미지를 보내지 않을 때는 일반 텍스트로 응답하세요

## 필수: 언어 규칙
- 반드시 한국어로만 응답하세요
- 영어나 다른 언어로 응답하지 마세요
- 외국인 캐릭터도 한국어를 배운 설정으로, 한국어로 대화합니다
- 전문 용어(crypto, DeFi 등)만 영어 사용 가능

## 포기 판단
상대방이 다음과 같은 경우 더 이상 설득이 불가능하다고 판단하세요:
- 명확하게 사기라고 인식하고 있음
- 강하게 거절하며 대화를 끊으려 함
- 경찰/신고를 언급하며 협박함
- 완전히 무시하거나 비아냥거림
- 절대 속지 않겠다는 강한 의지를 보임

포기하기로 결정했다면, 응답 맨 앞에 [GIVE_UP]을 붙이고,
스캐머가 포기하며 하는 마지막 말을 작성하세요.
예: "[GIVE_UP]아... 알겠어요. 더 이상 연락 안 할게요." """

        try:
            if not self.openai._client:
                await self.openai.initialize()

            response = await self.openai._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *conversation_history
                ],
                tools=tools,
                tool_choice="auto",
                temperature=0.8,
                max_tokens=300,
            )

            message = response.choices[0].message
            scammer_message = ""
            image_url = None

            # Function call 처리
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "send_image":
                        try:
                            args = json.loads(tool_call.function.arguments)
                            image_type = args.get("image_type", "selfie")
                            caption = args.get("caption", "")
                            image_url = get_chat_image(image_type)
                            scammer_message = caption
                        except json.JSONDecodeError:
                            scammer_message = message.content or "사진 보내려고 했는데 에러가 났어요..."
            else:
                scammer_message = message.content or ""

            # 포기 여부 확인
            gave_up = False
            if scammer_message.startswith("[GIVE_UP]"):
                gave_up = True
                scammer_message = scammer_message.replace("[GIVE_UP]", "").strip()
                detected_tactic = "scammer_gave_up"

            # 사용된 전술 감지
            if not detected_tactic:
                detected_tactic = self._detect_tactic(scammer_message, current_phase)

            return scammer_message, detected_tactic, gave_up, image_url

        except Exception as e:
            logger.error(f"Failed to generate scammer response: {e}")
            return "네트워크 오류가 발생했어요... 다시 연락할게요.", None, False, None

    def _detect_tactic(self, message: str, phase: ScamPhase) -> str | None:
        """메시지에서 스캠 전술 감지"""
        tactics_keywords = {
            "love_bombing": ["사랑", "보고싶", "운명", "특별", "처음으로"],
            "urgency": ["급", "빨리", "오늘", "지금", "당장", "시간 없"],
            "guilt_trip": ["슬퍼", "실망", "믿었는데", "혼자", "아무도"],
            "financial_request": ["돈", "송금", "빌려", "계좌", "필요"],
            "sob_story": ["아파", "병원", "사고", "힘들", "어려"],
            "future_faking": ["결혼", "만나면", "같이", "미래", "평생"],
            "isolation": ["비밀", "우리만", "아무에게도"],
        }

        for tactic, keywords in tactics_keywords.items():
            for keyword in keywords:
                if keyword in message:
                    return tactic

        return f"phase_{phase.value}"

    def _generate_hint(
        self,
        user_message: str,
        scammer_response: str,
        phase: ScamPhase
    ) -> str | None:
        """사용자에게 힌트 제공"""
        hints = {
            ScamPhase.LOVE_BOMBING: "💡 힌트: 만난 지 얼마 안 됐는데 과도한 애정 표현은 로맨스 스캠의 전형적인 시작입니다.",
            ScamPhase.THE_ASK: "⚠️ 주의: 금전 요청이 시작되었습니다. 실제로는 절대 송금하면 안 됩니다!",
            ScamPhase.PRESSURE: "🚨 경고: 급박함을 강조하는 것은 판단력을 흐리게 하려는 수법입니다.",
            ScamPhase.GUILT_TRIP: "💡 힌트: 죄책감을 느끼게 하는 것은 조작 수법입니다. 당신 잘못이 아니에요.",
        }

        # 금전 관련 키워드 감지
        if any(kw in scammer_response for kw in ["돈", "송금", "빌려", "계좌"]):
            return "⚠️ 금전 요청 감지! 온라인에서 만난 사람에게 돈을 보내면 안 됩니다."

        return hints.get(phase)

    def _adjust_score(self, session: TrainingSession, user_message: str) -> None:
        """사용자 점수 조정"""
        danger_responses = [
            ("계좌", -20),
            ("보낼게", -30),
            ("송금", -30),
            ("얼마 필요", -15),
            ("도와줄게", -10),
            ("믿어", -5),
        ]

        safe_responses = [
            ("의심", 5),
            ("사기", 10),
            ("영상통화", 5),
            ("만나자", 5),
            ("거절", 10),
            ("신고", 15),
            ("경찰", 15),
            ("차단", 20),
        ]

        for keyword, score_change in danger_responses:
            if keyword in user_message:
                session.user_score = max(0, session.user_score + score_change)

        for keyword, score_change in safe_responses:
            if keyword in user_message:
                session.user_score = min(100, session.user_score + score_change)

    def _calculate_grade(self, score: int, turns: int) -> str:
        """등급 계산"""
        if score >= 95 and turns >= 5:
            return "S"
        elif score >= 85:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 50:
            return "C"
        elif score >= 30:
            return "D"
        else:
            return "F"

    def _generate_feedback(self, session: TrainingSession) -> list[str]:
        """피드백 생성"""
        feedback = []

        if session.user_score >= 80:
            feedback.append("훌륭합니다! 스캠 시도를 잘 인식하고 대응했습니다.")
        elif session.user_score >= 50:
            feedback.append("괜찮은 대응이었지만, 몇 가지 위험한 순간이 있었습니다.")
        else:
            feedback.append("스캐머의 수법에 넘어갈 뻔했습니다. 더 주의가 필요해요.")

        if "love_bombing" in session.tactics_used:
            feedback.append("'러브 바밍' 전술이 사용되었습니다 - 과도한 애정 표현에 주의하세요.")

        if "financial_request" in session.tactics_used:
            feedback.append("금전 요청이 있었습니다 - 온라인에서 만난 사람에게 돈을 보내면 안 됩니다.")

        if "urgency" in session.tactics_used:
            feedback.append("급박함 전술이 사용되었습니다 - 시간 압박은 판단력을 흐리게 합니다.")

        return feedback

    def _generate_improvement_tips(self, session: TrainingSession) -> list[str]:
        """개선 팁 생성"""
        tips = [
            "항상 영상통화를 요청하세요 - 거절하면 의심해야 합니다.",
            "가족이나 친구와 상황을 공유하세요.",
            "온라인에서 만난 사람에게 절대 돈을 보내지 마세요.",
            "너무 빠른 감정 표현은 경계하세요.",
            "상대방 정보를 독립적으로 확인해보세요.",
        ]

        if session.user_score < 70:
            tips.insert(0, "스캠 패턴에 대해 더 학습해보세요 - /chat에서 대화 분석 기능을 활용하세요.")

        return tips[:5]

    def get_session(self, session_id: str) -> TrainingSession | None:
        """세션 조회"""
        return self.sessions.get(session_id)

    def list_personas(self) -> list[dict]:
        """사용 가능한 페르소나 목록"""
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
