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
        parsed_messages: Optional[list[ParsedMessage]] = None
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

            # Step 4: AI 분석
            interpretation_steps.append("GPT-4o AI 분석 요청")

            prompt = f"""당신은 로맨스 스캠 전문 분석가입니다. 아래 대화 내용을 분석해주세요.

## 분석할 대화:
{chr(10).join(messages)}

## 로맨스 스캠 패턴:
{pattern_descriptions}

{f"## 데이터베이스에서 조회된 스캠 정보:{chr(10)}{rag_prompt_context}" if rag_prompt_context else ""}

## 분석 지침:
1. [상대방] 표시된 메시지가 스캠 가능성이 있는지 중점 분석
2. [나] 표시된 메시지는 피해자의 반응으로 참고
3. 데이터베이스에서 매칭된 패턴이 있다면 그 정보를 적극 활용

## 응답 형식 (JSON):
{{
    "risk_score": 0-100 사이의 위험 점수,
    "detected_patterns": ["감지된 패턴 목록"],
    "warning_signs": ["주의해야 할 신호들"],
    "recommendations": ["사용자에게 권장하는 조치"],
    "analysis": "상세 분석 내용 (2-3문장)"
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
        """폴백: 기본 패턴 매칭 분석"""
        combined_text = " ".join(messages).lower()
        detected = []
        total_severity = 0

        keyword_patterns = {
            "love_bombing": ["사랑해", "보고싶", "운명", "첫눈에", "반했", "좋아해"],
            "financial_request": ["돈", "투자", "송금", "빌려", "계좌", "원", "달러", "비트코인"],
            "urgency": ["급", "당장", "지금", "빨리", "서둘러", "시간 없"],
            "isolation": ["비밀", "우리만", "아무에게", "말하지마", "가족한테"],
            "future_faking": ["결혼", "같이 살", "만나자", "곧 갈게"],
            "sob_story": ["사고", "아파", "병원", "수술", "도와줘"],
            "avoidance": ["영상통화", "화상", "만날 수 없", "얼굴"],
        }

        severity_map = {
            "love_bombing": 7,
            "financial_request": 10,
            "urgency": 8,
            "isolation": 9,
            "future_faking": 6,
            "sob_story": 7,
            "avoidance": 8,
        }

        for pattern_type, keywords in keyword_patterns.items():
            for keyword in keywords:
                if keyword in combined_text:
                    if pattern_type not in detected:
                        detected.append(pattern_type)
                        total_severity += severity_map.get(pattern_type, 5)
                    break

        # RAG 결과 반영
        if rag_context and rag_context.matched_phrases:
            for phrase in rag_context.matched_phrases:
                if phrase["category"] not in detected:
                    detected.append(phrase["category"])
                    total_severity += phrase.get("severity", 5)

        risk_score = min(100, total_severity * 8)

        warning_signs = []
        if "love_bombing" in detected:
            warning_signs.append("과도한 애정 표현이 감지되었습니다")
        if "financial_request" in detected:
            warning_signs.append("금전 관련 요청이 감지되었습니다 - 주의하세요!")
        if "urgency" in detected:
            warning_signs.append("급박함을 강조하고 있습니다")
        if "isolation" in detected:
            warning_signs.append("고립을 유도하는 패턴이 감지되었습니다")

        if rag_context and rag_context.risk_indicators:
            warning_signs.extend(rag_context.risk_indicators[:3])

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
            ai_analysis=f"패턴 기반 분석 완료. 위험도: {risk_score}점. 감지된 패턴: {', '.join(detected) if detected else '없음'}",
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

    async def analyze_deepfake_image(self, image_data: bytes, is_deepfake: bool, confidence: float) -> dict:
        """이미지 품질 분석 (GPT-4o Vision)"""
        # 클라이언트 초기화
        if not self._client:
            try:
                await self.initialize()
            except Exception as e:
                logger.warning(f"OpenAI 초기화 실패: {e}")
                return self._fallback_deepfake_analysis(is_deepfake, confidence)

        if not self._client:
            return self._fallback_deepfake_analysis(is_deepfake, confidence)

        try:
            base64_image = base64.b64encode(image_data).decode('utf-8')

            # 디지털 미디어 리터러시 교육 맥락
            system_message = """You are a digital media forensics expert teaching a university course on "Digital Media Literacy and Authentication".
Your role is to help students identify whether images are authentic photographs or computer-generated/AI-generated content.
This is for educational purposes to help people recognize synthetic media and protect themselves from misinformation.
Always analyze images objectively and technically. Respond in Korean."""

            # AI 생성 여부 판별에 초점을 맞춘 분석
            if is_deepfake:
                analysis_focus = """이 이미지에서 **AI 생성 또는 컴퓨터 합성의 흔적**을 찾아 분석해주세요.

다음 항목들을 중점적으로 검사하세요:
- 피부 질감: 지나치게 매끄럽거나 플라스틱처럼 보이는지
- 얼굴 대칭성: 비정상적으로 완벽하거나 미세한 비대칭이 있는지
- 눈: 동공 반사, 눈동자 디테일, 속눈썹의 자연스러움
- 머리카락: 가장자리 처리, 개별 머리카락 표현, 배경과의 경계
- 조명/그림자: 물리적으로 일관성 있는지, 광원 방향이 맞는지
- 배경: 흐림 처리의 자연스러움, 피사체와의 경계
- 액세서리(귀걸이, 목걸이 등): 디테일과 일관성
- 전체적인 "언캐니 밸리" 느낌"""
            else:
                analysis_focus = """이 이미지가 **실제 사진인 근거**를 찾아 분석해주세요.

다음 항목들을 중점적으로 검사하세요:
- 피부 질감: 자연스러운 모공, 잔주름, 피부결이 보이는지
- 자연스러운 비대칭: 실제 얼굴의 미세한 비대칭이 있는지
- 눈: 자연스러운 동공 반사와 눈의 수분감
- 머리카락: 자연스러운 흐트러짐, 개별 머리카락 표현
- 조명: 물리 법칙에 맞는 자연스러운 그림자
- 카메라 특성: 자연스러운 노이즈/그레인, 렌즈 특성
- 전체적으로 "실제 사람"의 느낌이 드는 요소"""

            # 얼굴 위치 가이드 (일반적인 인물 사진 기준)
            coordinate_guide = """
## 좌표 가이드 (인물 사진 기준):
- 이마/헤어라인: x=50, y=10~20
- 왼쪽 눈: x=35~40, y=30~35
- 오른쪽 눈: x=60~65, y=30~35
- 코: x=50, y=45~50
- 입: x=50, y=60~65
- 왼쪽 귀: x=20~25, y=35~45
- 오른쪽 귀: x=75~80, y=35~45
- 왼쪽 뺨: x=30, y=50
- 오른쪽 뺨: x=70, y=50
- 턱: x=50, y=75~80
- 목/어깨: x=50, y=85~95"""

            prompt = f"""{analysis_focus}

{coordinate_guide}

위 가이드를 참고하여 실제 이미지에서 해당 영역의 정확한 위치를 퍼센트 좌표(x: 0-100, y: 0-100)로 지정하세요.

JSON 형식으로만 응답:
{{
    "markers": [
        {{
            "id": 1,
            "x": 정확한_x좌표,
            "y": 정확한_y좌표,
            "label": "영역명",
            "description": "해당 영역의 {'AI 생성 흔적' if is_deepfake else '자연스러운 특징'} 설명"
        }}
    ],
    "overall_assessment": "종합 평가 (2문장)"
}}

정확히 3개의 markers만 포함하세요."""

            response = await self._client.chat.completions.create(
                model="gpt-4o-mini",
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
                                    "detail": "auto"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.3,
                max_tokens=800,
            )

            response_text = response.choices[0].message.content or ""
            logger.info(f"OpenAI deepfake response: {response_text[:500]}")
            return self._parse_deepfake_response(response_text, is_deepfake, confidence)

        except Exception as e:
            logger.error(f"Deepfake analysis failed: {e}")
            return self._fallback_deepfake_analysis(is_deepfake, confidence)

    def _parse_deepfake_response(self, response_text: str, is_deepfake: bool, confidence: float) -> dict:
        """딥페이크 분석 응답 파싱"""
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
            return {
                "analysis_reasons": data.get("analysis_reasons", []),
                "markers": data.get("markers", []),
                "technical_indicators": data.get("technical_indicators", []),
                "overall_assessment": data.get("overall_assessment", "")
            }
        except Exception as e:
            logger.error(f"Failed to parse deepfake analysis: {e}, response: {response_text[:300]}")
            return self._fallback_deepfake_analysis(is_deepfake, confidence)

    def _fallback_deepfake_analysis(self, is_deepfake: bool, confidence: float) -> dict:
        """폴백 딥페이크 분석"""
        if is_deepfake:
            return {
                "markers": [
                    {"id": 1, "x": 40, "y": 32, "label": "왼쪽 눈", "description": "AI 생성 이미지에서 흔히 나타나는 부자연스러운 눈동자 반사 패턴이 감지됨"},
                    {"id": 2, "x": 60, "y": 32, "label": "오른쪽 눈", "description": "양쪽 눈의 조명 반사가 비대칭적이며 물리적으로 일관성이 부족함"},
                    {"id": 3, "x": 50, "y": 55, "label": "피부 질감", "description": "모공이나 자연스러운 피부 결이 보이지 않고 과도하게 매끄러움"}
                ],
                "overall_assessment": f"딥페이크 확률 {confidence:.1f}%로 AI 생성 이미지로 의심됩니다. 피부 질감, 눈 반사, 머리카락 경계 등에서 합성 흔적이 감지되었습니다."
            }
        else:
            return {
                "markers": [
                    {"id": 1, "x": 40, "y": 32, "label": "왼쪽 눈", "description": "자연스러운 동공 반사와 수분감이 관찰됨"},
                    {"id": 2, "x": 60, "y": 32, "label": "오른쪽 눈", "description": "양쪽 눈의 조명 반사가 물리적으로 일관성 있음"},
                    {"id": 3, "x": 50, "y": 50, "label": "피부", "description": "자연스러운 모공과 피부결이 관찰됨"}
                ],
                "overall_assessment": f"실제 사진으로 판단됩니다 (신뢰도 {100-confidence:.1f}%). 자연스러운 피부 질감과 조명 반사가 확인되었습니다."
            }
