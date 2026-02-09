"""
OpenAI API 서비스
GPT-4o를 사용한 채팅 분석 및 스크린샷 분석
"""
import base64
import json
import logging
import re
from typing import Optional

from openai import AsyncOpenAI

from src.domain.chat import (
    ROMANCE_SCAM_PATTERNS,
    ChatAnalysisResult,
    RAGContext,
    ParsedMessage,
    MessageParser,
    MessageRole,
)
from src.infrastructure.ai import get_face_landmark_service
from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class OpenAIService:
    """OpenAI GPT 서비스"""

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.openai_api_key
        self._client: Optional[AsyncOpenAI] = None

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def initialize(self) -> None:
        """OpenAI 클라이언트 초기화"""
        if not self.is_configured():
            logger.warning("OpenAI API key not configured")
            return

        try:
            self._client = AsyncOpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            self._client = None

    async def analyze_chat(
        self,
        messages: list[str],
        rag_context: Optional[RAGContext] = None,
        parsed_messages: Optional[list[ParsedMessage]] = None,
        relationship_context: Optional[dict] = None
    ) -> ChatAnalysisResult:
        """채팅 메시지 분석"""
        interpretation_steps = []

        # 클라이언트 초기화
        if not self._client:
            try:
                await self.initialize()
            except Exception as e:
                logger.warning(f"OpenAI 초기화 실패, 폴백 사용: {e}")
                interpretation_steps.append("AI 서비스 연결 실패 - 패턴 매칭으로 분석")

        # 클라이언트가 없으면 폴백
        if not self._client:
            interpretation_steps.append("패턴 기반 분석 수행")
            result = self._fallback_analysis(messages, rag_context)
            result.parsed_messages = MessageParser.parse_messages(messages) if messages else None
            result.interpretation_steps = interpretation_steps
            return result

        try:
            # Step 1: 메시지 파싱 정보
            if parsed_messages:
                sender_count = sum(1 for p in parsed_messages if p.role.value == "sender")
                receiver_count = sum(1 for p in parsed_messages if p.role.value == "receiver")
                interpretation_steps.append(f"메시지 파싱: 상대방 {sender_count}개, 본인 {receiver_count}개 메시지 식별")

            # Step 2: RAG 조회 결과
            rag_prompt_context = ""
            if rag_context:
                if rag_context.matched_phrases:
                    interpretation_steps.append(f"RAG 조회: {len(rag_context.matched_phrases)}개 스캠 패턴 매칭")
                if rag_context.similar_cases:
                    interpretation_steps.append(f"유사 사례: {len(rag_context.similar_cases)}건의 스캠 사례 발견")
                rag_prompt_context = rag_context.to_prompt_context()

            # Step 3: 패턴 목록 생성
            pattern_descriptions = "\n".join([
                f"- {p.pattern_type}: {p.description} (예: {', '.join(p.examples[:2])})"
                for p in ROMANCE_SCAM_PATTERNS
            ])
            interpretation_steps.append("기본 패턴 매칭 수행")

            # Step 3.5: 관계/맥락 정보 처리
            relationship_prompt_context = ""
            if relationship_context:
                rel_info = relationship_context.get("relationship")
                ctx_info = relationship_context.get("context")
                protective = relationship_context.get("protective_factors", [])
                risk = relationship_context.get("risk_factors", [])

                parts = []
                if rel_info:
                    rel_type = rel_info.get("type", "unknown")
                    trust = rel_info.get("trust_level", 0)
                    interaction = rel_info.get("interaction_count", 0)
                    parts.append(f"- 관계 유형: {rel_type} (신뢰도: {trust:.0%}, 대화 횟수: {interaction}회)")
                    interpretation_steps.append(f"관계 정보: {rel_type}, 신뢰도 {trust:.0%}")

                if ctx_info:
                    ctx_type = ctx_info.get("type", "unknown")
                    ctx_keywords = ctx_info.get("keywords", [])
                    ctx_conf = ctx_info.get("confidence", 0)
                    parts.append(f"- 감지된 대화 맥락: {ctx_type} (키워드: {', '.join(ctx_keywords)}, 신뢰도: {ctx_conf:.0%})")
                    interpretation_steps.append(f"맥락 감지: {ctx_type} ({', '.join(ctx_keywords)})")

                if protective:
                    parts.append(f"- 보호 요소: {', '.join(protective)}")

                if risk:
                    parts.append(f"- 위험 요소: {', '.join(risk)}")

                if parts:
                    relationship_prompt_context = "## 사용자 관계 및 맥락 정보:\n" + "\n".join(parts)

            # Step 4: AI 분석
            interpretation_steps.append("GPT-4o AI 분석 요청")

            prompt = f"""당신은 대화 맥락 분석 전문가입니다. 아래 대화 내용을 분석해주세요.

## 1단계: 대화 맥락 판단 (가장 중요 - 먼저 수행)
다음 중 어떤 상황인지 먼저 판단하세요:
- **friend_casual**: 친구/지인 간 일상 대화 (게임, 내기, 장난, 송금 요청 등)
- **online_stranger**: 온라인에서 처음 만난 사람과의 대화
- **romance**: 로맨스/연애 맥락의 대화

## 친구 대화 신호 (위험도 대폭 낮춤 - 0~20점):
- 반말, 줄임말 사용: ㅋㅋ, ㅎㅎ, ㄱ?, ㄴㄴ, ㅇㅇ, ㄱㄱ, ㅇㅋ
- 게임/내기 맥락: 롤, 배그, 오버워치, 한판, 빵, 내기, 졌다, 이겼다
- 친구 간 금전 표현: "만원빵", "내놔", "한턱", "사줘", "입금해", "계좌 보내"
- 게임 내기 결과로 보이는 금전 요청
- 장난스러운 협박: "넌 뒤졌다", "죽었어", "각오해"

## 로맨스 스캠 신호 (위험도 높임 - 60~100점):
- 낯선 사람의 과도한 애정 표현 (운명, 소울메이트, 첫눈에 반했어)
- 긴급한 금전 요구 + 감정 조종 (나를 믿지 못하는 거야?)
- 만남 회피, 영상통화 거부 (보안상, 카메라 고장)
- 해외 체류 핑계 (군인, 석유 시추, UN)
- 투자 권유, 원금 보장 약속

## 가스라이팅/감정 조작 신호 (위험도 높임 - 70~100점):
- 현실 왜곡: "내가 언제 그랬어?", "그런 적 없어", "네가 기억을 잘못하는 거야"
- 감정 무효화: "너무 예민해", "별것도 아닌 걸로", "왜 그렇게 부정적이야"
- 죄책감 유발: "나를 사랑하면 당연히", "다 너를 위해서야", "내가 이렇게까지 하는데"
- 자존감 파괴: "나 아니면 누가 널 좋아하겠어", "넌 나 없으면 안 돼"
- 책임 전가: "네가 잘못한 거야", "네가 그렇게 만든 거야", "다 네 탓이야"

## 분석할 대화:
{chr(10).join(messages)}

## 알려진 스캠 패턴:
{pattern_descriptions}

{f"## 데이터베이스에서 조회된 정보:{chr(10)}{rag_prompt_context}" if rag_prompt_context else ""}

{relationship_prompt_context if relationship_prompt_context else ""}

## 분석 지침:
1. **맥락을 먼저 파악**: 친구 간 대화인지, 낯선 사람과의 대화인지 판단
2. **관계 정보 활용**: 위에 제공된 관계/맥락 정보가 있다면 적극 반영
3. 친구 간 게임 내기로 인한 계좌번호/송금 요청은 정상 (위험도 낮음)
4. [상대방] 메시지가 스캠 가능성이 있는지 분석
5. [나] 메시지는 참고용
6. 데이터베이스 매칭 패턴 활용

## 응답 형식 (JSON):
{{
    "context_type": "friend_casual|online_stranger|romance",
    "risk_score": 0-100 사이의 위험 점수,
    "detected_patterns": ["감지된 패턴 목록"],
    "warning_signs": ["주의해야 할 신호들"],
    "recommendations": ["사용자에게 권장하는 조치"],
    "analysis": "상세 분석 내용 (맥락 판단 근거 포함, 2-3문장)"
}}

JSON만 응답해주세요."""

            response = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )

            result = self._parse_response(response.choices[0].message.content or "")

            # 추가 정보 설정
            result.rag_context = rag_context
            result.parsed_messages = parsed_messages
            result.interpretation_steps = interpretation_steps + ["AI 분석 완료"]

            return result

        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            interpretation_steps.append(f"AI 분석 실패: {str(e)[:50]}")
            interpretation_steps.append("폴백 패턴 매칭 수행")
            result = self._fallback_analysis(messages, rag_context)
            result.parsed_messages = parsed_messages
            result.interpretation_steps = interpretation_steps
            return result

    async def analyze_chat_screenshot(
        self,
        image_data: bytes,
        rag_context: Optional[RAGContext] = None
    ) -> ChatAnalysisResult:
        """채팅 스크린샷 분석 (GPT-4o Vision)"""
        interpretation_steps = ["스크린샷 분석 시작"]

        # 클라이언트 초기화
        if not self._client:
            try:
                await self.initialize()
            except Exception as e:
                logger.warning(f"OpenAI 초기화 실패: {e}")

        if not self._client:
            interpretation_steps.append("AI 서비스 연결 실패")
            return ChatAnalysisResult.create(
                risk_score=0,
                detected_patterns=[],
                warning_signs=["AI 서비스를 사용할 수 없습니다"],
                recommendations=["텍스트로 대화 내용을 직접 입력해주세요"],
                ai_analysis="스크린샷 분석을 위해서는 AI 서비스가 필요합니다.",
                interpretation_steps=interpretation_steps
            )

        try:
            # 이미지를 base64로 인코딩
            base64_image = base64.b64encode(image_data).decode('utf-8')
            interpretation_steps.append("이미지 인코딩 완료")

            # RAG 컨텍스트
            rag_prompt_context = ""
            if rag_context:
                rag_prompt_context = rag_context.to_prompt_context()
                if rag_context.matched_phrases:
                    interpretation_steps.append(f"RAG 조회: {len(rag_context.matched_phrases)}개 패턴")

            # 패턴 목록
            pattern_descriptions = "\n".join([
                f"- {p.pattern_type}: {p.description} (예: {', '.join(p.examples[:2])})"
                for p in ROMANCE_SCAM_PATTERNS
            ])

            interpretation_steps.append("GPT-4o Vision AI 분석 요청")

            prompt = f"""이 이미지는 메신저/채팅 앱의 스크린샷입니다.
대화 내용을 읽고 로맨스 스캠 위험도를 분석해주세요.

## 로맨스 스캠 패턴:
{pattern_descriptions}

{f"## 데이터베이스에서 조회된 스캠 정보:{chr(10)}{rag_prompt_context}" if rag_prompt_context else ""}

## 분석 지침:
1. 스크린샷에서 대화 내용을 추출하세요
2. 발신자(상대방)와 수신자(본인)를 구분하세요 (보통 오른쪽이 본인, 왼쪽이 상대방)
3. 상대방의 메시지에서 스캠 패턴을 찾으세요
4. 전체적인 대화 맥락을 고려하세요

## 응답 형식 (JSON):
{{
    "risk_score": 0-100 사이의 위험 점수,
    "detected_patterns": ["감지된 패턴 목록"],
    "warning_signs": ["주의해야 할 신호들"],
    "recommendations": ["사용자에게 권장하는 조치"],
    "analysis": "상세 분석 내용 (2-3문장)",
    "extracted_messages": [
        {{"role": "sender", "content": "상대방 메시지"}},
        {{"role": "receiver", "content": "본인 메시지"}}
    ]
}}

JSON만 응답해주세요."""

            response = await self._client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.3,
                max_tokens=1500,
            )

            interpretation_steps.append("AI 분석 완료")

            result = self._parse_screenshot_response(response.choices[0].message.content or "")
            result.interpretation_steps = interpretation_steps
            result.rag_context = rag_context

            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Screenshot analysis failed: {error_msg}")
            interpretation_steps.append(f"분석 실패: {error_msg[:50]}")

            # API 할당량 초과 에러 처리
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                return ChatAnalysisResult.create(
                    risk_score=0,
                    detected_patterns=[],
                    warning_signs=["AI 서비스 사용량 제한에 도달했습니다"],
                    recommendations=[
                        "잠시 후 다시 시도해주세요",
                        "또는 대화 내용을 텍스트로 직접 입력해주세요"
                    ],
                    ai_analysis="API 사용량 제한으로 스크린샷 분석이 불가합니다.",
                    interpretation_steps=interpretation_steps
                )

            return ChatAnalysisResult.create(
                risk_score=0,
                detected_patterns=[],
                warning_signs=["스크린샷 분석 중 오류 발생"],
                recommendations=["다시 시도하거나 텍스트로 직접 입력해주세요"],
                ai_analysis="스크린샷 분석에 실패했습니다.",
                interpretation_steps=interpretation_steps
            )

    def _parse_response(self, response_text: str) -> ChatAnalysisResult:
        """응답 파싱"""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                raise ValueError("No JSON found in response")

            data = json.loads(json_match.group())

            return ChatAnalysisResult.create(
                risk_score=min(100, max(0, int(data.get("risk_score", 0)))),
                detected_patterns=data.get("detected_patterns", []),
                warning_signs=data.get("warning_signs", []),
                recommendations=data.get("recommendations", []),
                ai_analysis=data.get("analysis", "분석 결과를 가져올 수 없습니다.")
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse response: {e}")
            return ChatAnalysisResult.create(
                risk_score=0,
                detected_patterns=[],
                warning_signs=[],
                recommendations=["AI 분석에 실패했습니다. 다시 시도해주세요."],
                ai_analysis=response_text[:500] if response_text else "응답 없음"
            )

    def _parse_screenshot_response(self, response_text: str) -> ChatAnalysisResult:
        """스크린샷 분석 응답 파싱"""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                raise ValueError("No JSON found in response")

            data = json.loads(json_match.group())

            # 추출된 메시지를 ParsedMessage로 변환
            parsed_messages = None
            if "extracted_messages" in data:
                parsed_messages = [
                    ParsedMessage(
                        role=MessageRole.SENDER if m.get("role") == "sender" else MessageRole.RECEIVER,
                        content=m.get("content", ""),
                        original=m.get("content", "")
                    )
                    for m in data.get("extracted_messages", [])
                ]

            return ChatAnalysisResult.create(
                risk_score=min(100, max(0, int(data.get("risk_score", 0)))),
                detected_patterns=data.get("detected_patterns", []),
                warning_signs=data.get("warning_signs", []),
                recommendations=data.get("recommendations", []),
                ai_analysis=data.get("analysis", "분석 결과를 가져올 수 없습니다."),
                parsed_messages=parsed_messages
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse screenshot response: {e}")
            return ChatAnalysisResult.create(
                risk_score=0,
                detected_patterns=[],
                warning_signs=[],
                recommendations=["스크린샷 분석 결과를 파싱할 수 없습니다"],
                ai_analysis=response_text[:500] if response_text else "응답 없음"
            )

    def _fallback_analysis(
        self,
        messages: list[str],
        rag_context: Optional[RAGContext] = None
    ) -> ChatAnalysisResult:
        """폴백: 기본 패턴 매칭 분석 (맥락 기반 개선)"""
        combined_text = " ".join(messages).lower()
        detected = []
        total_severity = 0

        # 1단계: 친구 대화 맥락 감지
        friend_context_keywords = [
            "ㅋㅋ", "ㅎㅎ", "ㄱ?", "ㄱㄱ", "ㄴㄴ", "ㅇㅇ", "ㅇㅋ",
            "롤", "게임", "배그", "오버워치", "발로란트", "한판",
            "빵", "내기", "졌", "이겼", "만원빵", "내놔", "한턱",
            "뒤졌", "죽었", "각오해", "사줘"
        ]

        gaming_betting_keywords = [
            "롤", "게임", "배그", "한판", "빵", "내기", "졌", "이겼",
            "만원빵", "만원 빵", "오천빵", "내놔", "한턱"
        ]

        # 친구 대화 점수 계산
        friend_score = sum(1 for kw in friend_context_keywords if kw in combined_text)
        is_gaming_betting = any(kw in combined_text for kw in gaming_betting_keywords)

        # 친구 대화로 판단되면 위험도 대폭 낮춤
        is_friend_context = friend_score >= 2 or is_gaming_betting

        # 스캠 키워드 패턴
        keyword_patterns = {
            "love_bombing": ["사랑해", "보고싶", "운명", "첫눈에", "반했", "소울메이트"],
            "financial_request": ["투자", "비트코인", "코인", "원금 보장", "고수익"],
            "urgency": ["급해", "당장", "지금 바로", "빨리", "서둘러", "시간 없어"],
            "isolation": ["비밀", "우리만", "아무에게", "말하지마", "가족한테 말하면"],
            "future_faking": ["결혼하자", "같이 살자", "곧 만나자", "갈게"],
            "sob_story": ["사고가 났", "아파서", "병원비", "수술비", "도와줘"],
            "avoidance": ["영상통화 안", "화상 안", "만날 수 없", "카메라 고장"],
            "gaslighting": ["네가 잘못", "너무 예민", "내가 언제 그랬", "그런 적 없", "기억을 잘못", "나 아니면 누가", "넌 나 없으면"],
            "guilt_tripping": ["사랑하면 당연히", "다 너를 위해", "믿지 못하는 거", "이렇게까지 하는데", "다 네 탓"],
        }

        # 친구 맥락에서는 무시할 키워드 (계좌, 송금 등은 친구 내기일 수 있음)
        friend_safe_keywords = ["계좌", "송금", "입금", "돈", "원", "빌려"]

        severity_map = {
            "love_bombing": 7,
            "financial_request": 10,
            "urgency": 8,
            "isolation": 9,
            "future_faking": 6,
            "sob_story": 7,
            "avoidance": 8,
            "gaslighting": 9,
            "guilt_tripping": 8,
        }

        for pattern_type, keywords in keyword_patterns.items():
            for keyword in keywords:
                if keyword in combined_text:
                    if pattern_type not in detected:
                        detected.append(pattern_type)
                        total_severity += severity_map.get(pattern_type, 5)
                    break

        # 친구 맥락이 아닌 경우에만 금전 관련 키워드 체크
        if not is_friend_context:
            for keyword in friend_safe_keywords:
                if keyword in combined_text and "financial_request" not in detected:
                    detected.append("financial_request")
                    total_severity += 6  # 맥락 불명시 중간 정도의 위험도
                    break

        # RAG 결과 반영 (정상 패턴은 제외)
        if rag_context and rag_context.matched_phrases:
            for phrase in rag_context.matched_phrases:
                # 정상 패턴(is_safe=True)은 스킵
                if phrase.get("is_safe", False):
                    continue
                if phrase["category"] not in detected:
                    detected.append(phrase["category"])
                    total_severity += phrase.get("severity", 5)

        # 위험도 계산
        risk_score = min(100, total_severity * 8)

        # 친구 맥락이면 위험도 대폭 하향
        context_type = "friend_casual" if is_friend_context else "unknown"
        if is_friend_context:
            risk_score = max(0, int(risk_score * 0.2))  # 80% 감소
            # gaming/betting 맥락에서 financial_request는 제거
            if is_gaming_betting and "financial_request" in detected:
                detected.remove("financial_request")

        warning_signs = []
        if not is_friend_context:
            if "love_bombing" in detected:
                warning_signs.append("과도한 애정 표현이 감지되었습니다")
            if "financial_request" in detected:
                warning_signs.append("금전 관련 요청이 감지되었습니다 - 주의하세요!")
            if "urgency" in detected:
                warning_signs.append("급박함을 강조하고 있습니다")
            if "isolation" in detected:
                warning_signs.append("고립을 유도하는 패턴이 감지되었습니다")
            if "gaslighting" in detected:
                warning_signs.append("가스라이팅(감정 조작) 패턴이 감지되었습니다 - 상대방이 당신의 판단력을 흔들고 있습니다")
            if "guilt_tripping" in detected:
                warning_signs.append("죄책감을 유발하여 조종하려는 패턴이 감지되었습니다")
        else:
            warning_signs.append("친구/지인 간의 일상 대화로 판단됩니다")

        if rag_context and rag_context.risk_indicators and not is_friend_context:
            warning_signs.extend(rag_context.risk_indicators[:3])

        # 분석 결과 텍스트
        if is_friend_context:
            analysis_text = f"친구 대화 맥락 감지 (게임/내기: {is_gaming_betting}). 위험도: {risk_score}점. 일상적인 대화로 판단됩니다."
        else:
            analysis_text = f"패턴 기반 분석 완료. 위험도: {risk_score}점. 감지된 패턴: {', '.join(detected) if detected else '없음'}"

        return ChatAnalysisResult.create(
            risk_score=risk_score,
            detected_patterns=detected,
            warning_signs=warning_signs if warning_signs else ["패턴 기반 분석 결과입니다"],
            recommendations=[
                "상대방의 신원을 확인하세요",
                "금전 요청에 절대 응하지 마세요",
                "가족이나 친구와 상담하세요",
                "영상통화를 요청하세요"
            ] if risk_score > 30 else ["현재 특별한 위험 신호는 감지되지 않았습니다"],
            ai_analysis=analysis_text,
            rag_context=rag_context
        )

    async def generate_response(self, user_message: str, context: str = "") -> str:
        """일반 대화 응답 생성"""
        if not self._client:
            await self.initialize()

        if not self._client:
            return "죄송합니다, AI 서비스를 사용할 수 없습니다."

        try:
            prompt = f"""당신은 로맨스 스캠 예방 전문가 '러브가드'입니다.
사용자의 질문에 친절하고 전문적으로 답변해주세요.

{f"컨텍스트: {context}" if context else ""}

사용자: {user_message}"""

            response = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
            )

            return response.choices[0].message.content or "응답을 생성할 수 없습니다."

        except Exception as e:
            logger.error(f"OpenAI response generation failed: {e}")
            return "죄송합니다, 응답을 생성하는 중 오류가 발생했습니다."

    async def generate_text(self, prompt: str) -> str:
        """범용 텍스트 생성 (신고서 등)"""
        if not self._client:
            await self.initialize()
        if not self._client:
            return ""

        try:
            response = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=2000,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI text generation failed: {e}")
            return ""

    async def analyze_deepfake_image(self, image_data: bytes, is_deepfake: bool, confidence: float) -> dict:
        """이미지 딥페이크 분석 (얼굴 랜드마크 + GPT-4o Vision)"""
        # 1. 먼저 얼굴 랜드마크로 정확한 좌표 얻기
        face_landmark_service = get_face_landmark_service()
        markers = face_landmark_service.get_analysis_markers(image_data, is_deepfake, count=3)
        logger.info(f"Face landmarks detected: {markers}")

        # 클라이언트 초기화
        if not self._client:
            try:
                await self.initialize()
            except Exception as e:
                logger.warning(f"OpenAI 초기화 실패: {e}")
                return self._fallback_deepfake_analysis_with_markers(markers, is_deepfake, confidence)

        if not self._client:
            return self._fallback_deepfake_analysis_with_markers(markers, is_deepfake, confidence)

        try:
            base64_image = base64.b64encode(image_data).decode('utf-8')

            # 시스템 메시지
            system_message = """You are a digital media forensics expert.
Your role is to analyze specific facial regions for signs of AI generation or authenticity.
Respond in Korean with technical but understandable explanations."""

            # 마커 위치 정보 생성
            marker_info = "\n".join([
                f"- 위치 {m['id']}: {m['label']} (x={m['x']}%, y={m['y']}%)"
                for m in markers
            ])

            if is_deepfake:
                analysis_type = "AI 생성/합성 흔적"
                focus_points = """각 위치에서 다음을 확인하세요:
- 눈: 동공 반사 패턴, 눈동자 디테일, 속눈썹 자연스러움
- 피부/볼: 질감의 자연스러움, 모공 유무, 과도한 매끄러움
- 입: 입술 경계선, 치아 디테일
- 기타: 해당 부위의 AI 생성 특징"""
            else:
                analysis_type = "자연스러운 특징"
                focus_points = """각 위치에서 다음을 확인하세요:
- 눈: 자연스러운 동공 반사, 수분감
- 피부/볼: 자연스러운 모공, 피부결
- 입: 자연스러운 입술 질감
- 기타: 실제 사진의 특징"""

            prompt = f"""이 이미지의 다음 위치들을 분석해주세요:

{marker_info}

각 위치에서 **{analysis_type}**을 찾아 설명해주세요.

{focus_points}

JSON 형식으로만 응답:
{{
    "descriptions": {{
        "1": "위치 1에 대한 분석 설명 (1-2문장)",
        "2": "위치 2에 대한 분석 설명 (1-2문장)",
        "3": "위치 3에 대한 분석 설명 (1-2문장)"
    }},
    "overall_assessment": "종합 평가 (2문장)"
}}"""

            response = await self._client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.3,
                max_tokens=600,
            )

            response_text = response.choices[0].message.content or ""
            logger.info(f"OpenAI deepfake response: {response_text[:500]}")

            # 응답에서 설명만 파싱하여 마커에 추가
            return self._parse_deepfake_descriptions(response_text, markers, is_deepfake, confidence)

        except Exception as e:
            logger.error(f"Deepfake analysis failed: {e}")
            return self._fallback_deepfake_analysis_with_markers(markers, is_deepfake, confidence)

    def _parse_deepfake_descriptions(
        self,
        response_text: str,
        markers: list[dict],
        is_deepfake: bool,
        confidence: float
    ) -> dict:
        """GPT 응답에서 설명만 파싱하여 마커에 추가"""
        try:
            # markdown 코드 블록 제거
            cleaned = response_text
            if "```json" in cleaned:
                cleaned = re.sub(r'```json\s*', '', cleaned)
                cleaned = re.sub(r'```\s*$', '', cleaned)
            elif "```" in cleaned:
                cleaned = re.sub(r'```\s*', '', cleaned)

            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if not json_match:
                logger.error(f"No JSON found in response: {response_text[:200]}")
                raise ValueError("No JSON found")

            data = json.loads(json_match.group())
            descriptions = data.get("descriptions", {})

            # 마커에 설명 추가
            for marker in markers:
                marker_id = str(marker["id"])
                if marker_id in descriptions:
                    marker["description"] = descriptions[marker_id]
                else:
                    # 기본 설명
                    marker["description"] = self._get_default_description(
                        marker["label"], is_deepfake
                    )

            return {
                "markers": markers,
                "overall_assessment": data.get("overall_assessment", "")
            }
        except Exception as e:
            logger.error(f"Failed to parse descriptions: {e}, response: {response_text[:300]}")
            return self._fallback_deepfake_analysis_with_markers(markers, is_deepfake, confidence)

    def _get_default_description(self, label: str, is_deepfake: bool) -> str:
        """부위별 기본 설명"""
        if is_deepfake:
            defaults = {
                "왼쪽 눈": "부자연스러운 눈동자 반사 패턴이 감지됨",
                "오른쪽 눈": "조명 반사가 물리적으로 일관성이 부족함",
                "왼쪽 볼": "피부 질감이 과도하게 매끄러움",
                "오른쪽 볼": "자연스러운 피부결이 보이지 않음",
                "입": "입술 경계선이 부자연스러움",
                "코": "음영 처리가 물리적으로 맞지 않음",
                "이마": "피부 질감의 디테일이 부족함",
                "턱": "얼굴 윤곽선이 부자연스러움",
            }
        else:
            defaults = {
                "왼쪽 눈": "자연스러운 동공 반사와 수분감이 관찰됨",
                "오른쪽 눈": "자연스러운 눈의 디테일이 확인됨",
                "왼쪽 볼": "자연스러운 피부결과 모공이 보임",
                "오른쪽 볼": "실제 피부의 자연스러운 질감",
                "입": "자연스러운 입술 질감",
                "코": "자연스러운 음영과 하이라이트",
                "이마": "자연스러운 피부 디테일",
                "턱": "자연스러운 얼굴 윤곽",
            }
        return defaults.get(label, "분석 완료")

    def _fallback_deepfake_analysis_with_markers(
        self,
        markers: list[dict],
        is_deepfake: bool,
        confidence: float
    ) -> dict:
        """마커 좌표를 유지하면서 기본 설명 추가"""
        for marker in markers:
            if not marker.get("description"):
                marker["description"] = self._get_default_description(
                    marker["label"], is_deepfake
                )

        if is_deepfake:
            overall = f"딥페이크 확률 {confidence:.1f}%로 AI 생성 이미지로 의심됩니다."
        else:
            overall = f"실제 사진으로 판단됩니다 (신뢰도 {100-confidence:.1f}%)."

        return {
            "markers": markers,
            "overall_assessment": overall
        }

    def _fallback_deepfake_analysis(self, is_deepfake: bool, confidence: float) -> dict:
        """폴백 딥페이크 분석"""
        if is_deepfake:
            return {
                "markers": [
                    {"id": 1, "x": 38, "y": 42, "label": "왼쪽 눈", "description": "AI 생성 이미지에서 흔히 나타나는 부자연스러운 눈동자 반사 패턴이 감지됨"},
                    {"id": 2, "x": 62, "y": 42, "label": "오른쪽 눈", "description": "양쪽 눈의 조명 반사가 비대칭적이며 물리적으로 일관성이 부족함"},
                    {"id": 3, "x": 50, "y": 58, "label": "피부 질감", "description": "모공이나 자연스러운 피부 결이 보이지 않고 과도하게 매끄러움"}
                ],
                "overall_assessment": f"딥페이크 확률 {confidence:.1f}%로 AI 생성 이미지로 의심됩니다. 피부 질감, 눈 반사, 머리카락 경계 등에서 합성 흔적이 감지되었습니다."
            }
        else:
            return {
                "markers": [
                    {"id": 1, "x": 38, "y": 42, "label": "왼쪽 눈", "description": "자연스러운 동공 반사와 수분감이 관찰됨"},
                    {"id": 2, "x": 62, "y": 42, "label": "오른쪽 눈", "description": "양쪽 눈의 조명 반사가 물리적으로 일관성 있음"},
                    {"id": 3, "x": 50, "y": 55, "label": "피부", "description": "자연스러운 모공과 피부결이 관찰됨"}
                ],
                "overall_assessment": f"실제 사진으로 판단됩니다 (신뢰도 {100-confidence:.1f}%). 자연스러운 피부 질감과 조명 반사가 확인되었습니다."
            }

    async def analyze_deepfake_with_markers(
        self,
        image_data: bytes,
        markers: list[dict],
        is_deepfake: bool,
        confidence: float
    ) -> dict:
        """EfficientViT에서 감지된 마커에 대한 설명 생성"""
        if not self._client:
            try:
                await self.initialize()
            except Exception as e:
                logger.warning(f"OpenAI 초기화 실패: {e}")
                return self._fallback_marker_descriptions(markers, is_deepfake, confidence)

        if not self._client:
            return self._fallback_marker_descriptions(markers, is_deepfake, confidence)

        try:
            base64_image = base64.b64encode(image_data).decode('utf-8')

            system_message = """You are a digital media forensics expert.
Analyze the marked regions in the image and explain what you observe.
Respond in Korean with clear, technical explanations."""

            # 마커 정보
            marker_info = "\n".join([
                f"- 영역 {m['id']}: {m['label']} (위치: x={m['x']}%, y={m['y']}%, 의심 강도: {m.get('intensity', 0):.2f})"
                for m in markers
            ])

            if is_deepfake:
                analysis_type = "AI 생성/합성 흔적"
            else:
                analysis_type = "자연스러운 특징"

            prompt = f"""딥페이크 탐지 AI가 다음 영역들을 {'' if is_deepfake else '정상으로'} 표시했습니다:

{marker_info}

각 영역에서 **{analysis_type}**을 분석해주세요.
이미지를 직접 보고 해당 위치에서 관찰되는 특징을 설명해주세요.

JSON 형식으로만 응답:
{{
    "descriptions": {{
        "1": "영역 1에 대한 분석 (1-2문장)",
        "2": "영역 2에 대한 분석 (1-2문장)",
        "3": "영역 3에 대한 분석 (1-2문장)"
    }},
    "overall_assessment": "종합 평가 (2문장)"
}}"""

            response = await self._client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.3,
                max_tokens=600,
            )

            response_text = response.choices[0].message.content or ""
            logger.info(f"OpenAI marker analysis response: {response_text[:500]}")

            # JSON 파싱
            cleaned = response_text
            if "```json" in cleaned:
                cleaned = re.sub(r'```json\s*', '', cleaned)
                cleaned = re.sub(r'```\s*$', '', cleaned)
            elif "```" in cleaned:
                cleaned = re.sub(r'```\s*', '', cleaned)

            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if not json_match:
                raise ValueError("No JSON found")

            data = json.loads(json_match.group())
            return {
                "descriptions": data.get("descriptions", {}),
                "overall_assessment": data.get("overall_assessment", "")
            }

        except Exception as e:
            logger.error(f"Marker analysis failed: {e}")
            return self._fallback_marker_descriptions(markers, is_deepfake, confidence)

    def _fallback_marker_descriptions(
        self,
        markers: list[dict],
        is_deepfake: bool,
        confidence: float
    ) -> dict:
        """마커 설명 폴백"""
        descriptions = {}
        for marker in markers:
            marker_id = str(marker["id"])
            descriptions[marker_id] = self._get_default_description(
                marker.get("label", "영역"), is_deepfake
            )

        if is_deepfake:
            overall = f"딥페이크 확률 {confidence:.1f}%로 AI 생성 이미지로 의심됩니다."
        else:
            overall = f"실제 사진으로 판단됩니다 (신뢰도 {100-confidence:.1f}%)."

        return {
            "descriptions": descriptions,
            "overall_assessment": overall
        }
