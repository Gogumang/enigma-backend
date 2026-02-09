from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re


class RiskCategory(Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"


class MessageRole(Enum):
    """메시지 역할"""
    SENDER = "sender"      # 분석 대상 (상대방)
    RECEIVER = "receiver"  # 사용자 본인
    UNKNOWN = "unknown"


@dataclass
class ParsedMessage:
    """파싱된 메시지"""
    role: MessageRole
    content: str
    original: str


@dataclass
class ChatPattern:
    """채팅 패턴 값 객체"""
    pattern_type: str
    description: str
    severity: int  # 1-10
    examples: list[str]


@dataclass
class RAGContext:
    """RAG 조회 결과 컨텍스트"""
    matched_phrases: list[dict] = field(default_factory=list)
    similar_cases: list[dict] = field(default_factory=list)
    risk_indicators: list[str] = field(default_factory=list)
    total_reports: int = 0

    def to_prompt_context(self) -> str:
        """프롬프트에 포함할 컨텍스트 생성 (위험/안전 패턴 구분)"""
        if not self.matched_phrases and not self.similar_cases:
            return ""

        context_parts = []

        # 위험 패턴과 안전 패턴 분리
        risk_phrases = [p for p in self.matched_phrases if not p.get('is_safe', False)]
        safe_phrases = [p for p in self.matched_phrases if p.get('is_safe', False)]

        # 안전 패턴 (친구 대화) 먼저 표시 - AI가 맥락을 먼저 파악하도록
        if safe_phrases:
            safe_text = "\n".join([
                f"- '{p['text']}' (유형: {p['category']}, 유사도: {p.get('similarity', 0):.0f}%) - {p.get('description', '')}"
                for p in safe_phrases[:5]
            ])
            context_parts.append(f"## ✓ 매칭된 정상/친구 대화 패턴 (위험도 낮음):\n{safe_text}")

        # 위험 패턴
        if risk_phrases:
            risk_text = "\n".join([
                f"- '{p['text']}' (카테고리: {p['category']}, 위험도: {p['severity']}/10, 유사도: {p.get('similarity', 0):.0f}%)"
                for p in risk_phrases[:5]
            ])
            context_parts.append(f"## ⚠ 매칭된 스캠 패턴:\n{risk_text}")

        if self.similar_cases:
            cases_text = "\n".join([
                f"- {c['title']}: {c['description'][:100]}... (피해액: {c['damage_amount']:,}원)"
                for c in self.similar_cases[:3]
            ])
            context_parts.append(f"## 유사 스캠 사례:\n{cases_text}")

        if self.risk_indicators:
            indicators_text = "\n".join([f"- {i}" for i in self.risk_indicators[:5]])
            context_parts.append(f"## 위험 지표:\n{indicators_text}")

        return "\n\n".join(context_parts)


class MessageParser:
    """메시지 파싱 유틸리티"""

    # 발신자 표시 패턴
    SENDER_PATTERNS = [
        r'^(상대방|그\s*사람|스캐머|사기꾼|상대|그|그녀|그남자|그여자)\s*[:：]\s*',
        r'^\[(상대방|상대|그|그녀)\]\s*',
        r'^(A|상대)\s*[:：]\s*',
    ]

    # 수신자(본인) 표시 패턴
    RECEIVER_PATTERNS = [
        r'^(나|본인|저|내가|나는|me|나:)\s*[:：]\s*',
        r'^\[(나|본인|저|me)\]\s*',
        r'^(B|나)\s*[:：]\s*',
    ]

    @classmethod
    def parse_messages(cls, messages: list[str]) -> list[ParsedMessage]:
        """메시지 목록을 파싱하여 역할 구분"""
        parsed = []

        for msg in messages:
            msg = msg.strip()
            if not msg:
                continue

            role = MessageRole.UNKNOWN
            content = msg

            # 발신자 패턴 확인
            for pattern in cls.SENDER_PATTERNS:
                match = re.match(pattern, msg, re.IGNORECASE)
                if match:
                    role = MessageRole.SENDER
                    content = msg[match.end():].strip()
                    break

            # 수신자 패턴 확인
            if role == MessageRole.UNKNOWN:
                for pattern in cls.RECEIVER_PATTERNS:
                    match = re.match(pattern, msg, re.IGNORECASE)
                    if match:
                        role = MessageRole.RECEIVER
                        content = msg[match.end():].strip()
                        break

            parsed.append(ParsedMessage(
                role=role,
                content=content,
                original=msg
            ))

        return parsed

    @classmethod
    def get_sender_messages(cls, parsed: list[ParsedMessage]) -> list[str]:
        """발신자(상대방) 메시지만 추출"""
        return [p.content for p in parsed if p.role == MessageRole.SENDER]

    @classmethod
    def get_formatted_conversation(cls, parsed: list[ParsedMessage]) -> str:
        """역할 표시된 대화 형식으로 변환"""
        lines = []
        for p in parsed:
            role_label = {
                MessageRole.SENDER: "[상대방]",
                MessageRole.RECEIVER: "[나]",
                MessageRole.UNKNOWN: "[불명]"
            }[p.role]
            lines.append(f"{role_label} {p.content}")
        return "\n".join(lines)


@dataclass
class ChatAnalysisResult:
    """채팅 분석 결과 엔티티"""
    risk_score: int  # 0-100
    risk_category: RiskCategory
    detected_patterns: list[str]
    warning_signs: list[str]
    recommendations: list[str]
    ai_analysis: str
    analyzed_at: datetime
    # 추가 분석 정보
    rag_context: RAGContext | None = None
    parsed_messages: list[ParsedMessage] | None = None
    interpretation_steps: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        risk_score: int,
        detected_patterns: list[str],
        warning_signs: list[str],
        recommendations: list[str],
        ai_analysis: str,
        rag_context: RAGContext | None = None,
        parsed_messages: list[ParsedMessage] | None = None,
        interpretation_steps: list[str] | None = None
    ) -> "ChatAnalysisResult":
        # 위험 카테고리 결정
        if risk_score >= 70:
            category = RiskCategory.DANGEROUS
        elif risk_score >= 40:
            category = RiskCategory.SUSPICIOUS
        else:
            category = RiskCategory.SAFE

        return cls(
            risk_score=risk_score,
            risk_category=category,
            detected_patterns=detected_patterns,
            warning_signs=warning_signs,
            recommendations=recommendations,
            ai_analysis=ai_analysis,
            analyzed_at=datetime.now(),
            rag_context=rag_context,
            parsed_messages=parsed_messages,
            interpretation_steps=interpretation_steps or []
        )

    def to_dict(self) -> dict:
        result = {
            "risk_score": self.risk_score,
            "risk_category": self.risk_category.value,
            "detected_patterns": self.detected_patterns,
            "warning_signs": self.warning_signs,
            "recommendations": self.recommendations,
            "ai_analysis": self.ai_analysis,
            "analyzed_at": self.analyzed_at.isoformat(),
            "interpretation_steps": self.interpretation_steps,
        }

        if self.rag_context:
            result["rag_context"] = {
                "matched_phrases": self.rag_context.matched_phrases,
                "similar_cases": self.rag_context.similar_cases,
                "risk_indicators": self.rag_context.risk_indicators,
                "total_reports": self.rag_context.total_reports,
            }

        if self.parsed_messages:
            result["parsed_messages"] = [
                {
                    "role": p.role.value,
                    "content": p.content,
                }
                for p in self.parsed_messages
            ]

        return result


# 로맨스 스캠 패턴 정의
ROMANCE_SCAM_PATTERNS = [
    ChatPattern(
        pattern_type="love_bombing",
        description="과도한 애정 표현",
        severity=7,
        examples=["너무 사랑해", "운명이야", "첫눈에 반했어"]
    ),
    ChatPattern(
        pattern_type="financial_request",
        description="금전 요청",
        severity=10,
        examples=["돈 빌려줘", "투자해줘", "송금해줘"]
    ),
    ChatPattern(
        pattern_type="urgency",
        description="긴급성 강조",
        severity=8,
        examples=["지금 당장", "급해", "시간이 없어"]
    ),
    ChatPattern(
        pattern_type="isolation",
        description="고립 유도",
        severity=9,
        examples=["우리 둘만의 비밀", "아무에게도 말하지마", "가족한테 말하면 안돼"]
    ),
    ChatPattern(
        pattern_type="future_faking",
        description="미래 약속",
        severity=6,
        examples=["결혼하자", "같이 살자", "곧 만나자"]
    ),
    ChatPattern(
        pattern_type="sob_story",
        description="불쌍한 이야기",
        severity=7,
        examples=["사고가 났어", "아파", "돈이 없어"]
    ),
    ChatPattern(
        pattern_type="gaslighting",
        description="가스라이팅/감정 조작",
        severity=9,
        examples=["네가 잘못한 거야", "너무 예민해", "내가 언제 그랬어", "다 너를 위해서야"]
    ),
    ChatPattern(
        pattern_type="guilt_tripping",
        description="죄책감 유발",
        severity=8,
        examples=["나를 사랑하면 당연히", "나 때문에 힘들어", "믿지 못하는 거야?", "내가 이렇게까지 하는데"]
    ),
]
