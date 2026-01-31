"""
스캠 패턴 샘플 데이터 시드
실제 로맨스 스캠에서 사용되는 문구와 사례들
"""
import asyncio
from datetime import datetime, timedelta
import random

from .scam_pattern_repository import ScamPatternRepository, ScamPhrase, ScamCase


# ==================== 스캠 문구 데이터 ====================

SCAM_PHRASES = [
    # 러브바밍 (Love Bombing)
    ScamPhrase(
        id="lb_001",
        text="사랑해",
        category="love_bombing",
        severity=6,
        usage_count=892,
        examples=["사랑해 자기야", "너무 사랑해", "많이 사랑해"]
    ),
    ScamPhrase(
        id="lb_002",
        text="운명",
        category="love_bombing",
        severity=7,
        usage_count=456,
        examples=["우리는 운명이야", "운명적인 만남", "운명처럼 만났어"]
    ),
    ScamPhrase(
        id="lb_003",
        text="첫눈에 반했어",
        category="love_bombing",
        severity=8,
        usage_count=234,
        examples=["첫눈에 반했어", "사진 보고 반했어", "프로필 보고 반했어"]
    ),
    ScamPhrase(
        id="lb_004",
        text="보고싶어",
        category="love_bombing",
        severity=5,
        usage_count=1205,
        examples=["보고싶어", "너무 보고싶다", "빨리 보고싶어"]
    ),
    ScamPhrase(
        id="lb_005",
        text="평생 함께",
        category="love_bombing",
        severity=8,
        usage_count=321,
        examples=["평생 함께하고 싶어", "평생 너만 사랑할게", "평생 행복하게 해줄게"]
    ),

    # 금전 요청 (Financial Request)
    ScamPhrase(
        id="fr_001",
        text="투자",
        category="financial_request",
        severity=10,
        usage_count=1567,
        examples=["좋은 투자 기회가 있어", "같이 투자하자", "투자해서 돈 벌자"]
    ),
    ScamPhrase(
        id="fr_002",
        text="송금",
        category="financial_request",
        severity=10,
        usage_count=1234,
        examples=["돈 좀 송금해줘", "계좌로 보내줘", "송금 부탁해"]
    ),
    ScamPhrase(
        id="fr_003",
        text="빌려줘",
        category="financial_request",
        severity=10,
        usage_count=987,
        examples=["돈 좀 빌려줘", "잠깐만 빌려줘", "급하게 필요해서"]
    ),
    ScamPhrase(
        id="fr_004",
        text="비트코인",
        category="financial_request",
        severity=10,
        usage_count=2341,
        examples=["비트코인 투자", "코인으로 돈 벌자", "비트코인 좋은 기회"]
    ),
    ScamPhrase(
        id="fr_005",
        text="수익",
        category="financial_request",
        severity=9,
        usage_count=1876,
        examples=["수익이 엄청나", "100% 수익 보장", "매일 수익이 나와"]
    ),
    ScamPhrase(
        id="fr_006",
        text="계좌",
        category="financial_request",
        severity=9,
        usage_count=1432,
        examples=["계좌번호 알려줘", "내 계좌로 보내줘", "계좌 확인해봐"]
    ),

    # 긴급성 (Urgency)
    ScamPhrase(
        id="ur_001",
        text="급해",
        category="urgency",
        severity=8,
        usage_count=876,
        examples=["너무 급해", "급하게 필요해", "급한 일이 생겼어"]
    ),
    ScamPhrase(
        id="ur_002",
        text="지금 당장",
        category="urgency",
        severity=9,
        usage_count=654,
        examples=["지금 당장 필요해", "지금 바로 보내줘", "지금 아니면 안돼"]
    ),
    ScamPhrase(
        id="ur_003",
        text="시간이 없어",
        category="urgency",
        severity=8,
        usage_count=543,
        examples=["시간이 없어", "오늘까지야", "내일이면 늦어"]
    ),
    ScamPhrase(
        id="ur_004",
        text="마감",
        category="urgency",
        severity=8,
        usage_count=432,
        examples=["투자 마감이야", "오늘 마감이야", "마감 전에 해야해"]
    ),

    # 고립 유도 (Isolation)
    ScamPhrase(
        id="is_001",
        text="비밀",
        category="isolation",
        severity=9,
        usage_count=765,
        examples=["우리만의 비밀", "비밀로 해줘", "아무에게도 말하지마"]
    ),
    ScamPhrase(
        id="is_002",
        text="가족한테 말하지마",
        category="isolation",
        severity=10,
        usage_count=432,
        examples=["가족한테 말하지마", "부모님한테 비밀로", "친구한테 말하면 안돼"]
    ),
    ScamPhrase(
        id="is_003",
        text="우리 둘만",
        category="isolation",
        severity=8,
        usage_count=567,
        examples=["우리 둘만 알자", "우리끼리만", "둘만의 비밀"]
    ),

    # 회피 행동 (Avoidance)
    ScamPhrase(
        id="av_001",
        text="영상통화 안돼",
        category="avoidance",
        severity=9,
        usage_count=1234,
        examples=["영상통화는 좀", "화상통화 어려워", "영상은 못해"]
    ),
    ScamPhrase(
        id="av_002",
        text="만날 수 없어",
        category="avoidance",
        severity=8,
        usage_count=876,
        examples=["지금은 못 만나", "만나기 어려워", "나중에 만나자"]
    ),
    ScamPhrase(
        id="av_003",
        text="해외에 있어",
        category="avoidance",
        severity=7,
        usage_count=1543,
        examples=["지금 해외에 있어", "미국에서 일해", "군인이라 해외 파병중"]
    ),

    # 동정심 유발 (Sob Story)
    ScamPhrase(
        id="ss_001",
        text="사고가 났어",
        category="sob_story",
        severity=9,
        usage_count=654,
        examples=["교통사고 났어", "사고로 다쳤어", "사고가 생겼어"]
    ),
    ScamPhrase(
        id="ss_002",
        text="병원",
        category="sob_story",
        severity=8,
        usage_count=876,
        examples=["병원에 입원했어", "수술해야해", "병원비가 없어"]
    ),
    ScamPhrase(
        id="ss_003",
        text="도와줘",
        category="sob_story",
        severity=7,
        usage_count=1234,
        examples=["제발 도와줘", "나만 도와줄 수 있어", "도움이 필요해"]
    ),
    ScamPhrase(
        id="ss_004",
        text="아파",
        category="sob_story",
        severity=7,
        usage_count=543,
        examples=["많이 아파", "아프다", "몸이 안좋아"]
    ),

    # 미래 약속 (Future Faking)
    ScamPhrase(
        id="ff_001",
        text="결혼하자",
        category="future_faking",
        severity=8,
        usage_count=987,
        examples=["결혼하자", "너랑 결혼하고 싶어", "결혼해줘"]
    ),
    ScamPhrase(
        id="ff_002",
        text="같이 살자",
        category="future_faking",
        severity=7,
        usage_count=765,
        examples=["같이 살자", "동거하자", "한국 가서 같이 살자"]
    ),
    ScamPhrase(
        id="ff_003",
        text="곧 만나",
        category="future_faking",
        severity=6,
        usage_count=1432,
        examples=["곧 만나러 갈게", "곧 한국 갈게", "조금만 기다려"]
    ),

    # 신원 사칭 (Identity)
    ScamPhrase(
        id="id_001",
        text="군인",
        category="identity",
        severity=8,
        usage_count=2134,
        examples=["미군이야", "군인이라 힘들어", "해외 파병중"]
    ),
    ScamPhrase(
        id="id_002",
        text="의사",
        category="identity",
        severity=7,
        usage_count=876,
        examples=["의사야", "UN 의사", "국제기구 의사"]
    ),
    ScamPhrase(
        id="id_003",
        text="사업가",
        category="identity",
        severity=7,
        usage_count=1234,
        examples=["사업을 해", "CEO야", "무역업 해"]
    ),
]

# ==================== 스캠 사례 데이터 ====================

SCAM_CASES = [
    ScamCase(
        id="case_001",
        title="미군 사칭 로맨스 스캠",
        description="미군 장교를 사칭하여 6개월간 연애 후 귀국 비용, 세관 비용 명목으로 5000만원 편취",
        phrases_used=["lb_001", "lb_002", "fr_002", "av_003", "id_001"],
        damage_amount=50000000,
        reported_at=datetime.now() - timedelta(days=30),
        platform="instagram"
    ),
    ScamCase(
        id="case_002",
        title="비트코인 투자 유도 스캠",
        description="온라인에서 만나 연인 관계 발전 후 비트코인 투자 유도, 3000만원 피해",
        phrases_used=["lb_001", "lb_004", "fr_001", "fr_004", "fr_005"],
        damage_amount=30000000,
        reported_at=datetime.now() - timedelta(days=45),
        platform="kakao"
    ),
    ScamCase(
        id="case_003",
        title="병원비 명목 금전 요구",
        description="3개월 온라인 연애 후 갑작스러운 사고로 병원비 필요하다며 2000만원 요구",
        phrases_used=["lb_001", "lb_005", "ss_001", "ss_002", "ur_001"],
        damage_amount=20000000,
        reported_at=datetime.now() - timedelta(days=15),
        platform="telegram"
    ),
    ScamCase(
        id="case_004",
        title="세관 비용 요구 스캠",
        description="선물 보내준다며 세관 통과 비용 요구, 1500만원 피해",
        phrases_used=["lb_001", "ff_001", "fr_002", "ur_002", "is_001"],
        damage_amount=15000000,
        reported_at=datetime.now() - timedelta(days=60),
        platform="facebook"
    ),
    ScamCase(
        id="case_005",
        title="긴급 송금 요청 스캠",
        description="영상통화 회피하며 급하게 돈 필요하다고 반복 요청, 800만원 피해",
        phrases_used=["lb_004", "av_001", "ur_001", "fr_003", "ss_003"],
        damage_amount=8000000,
        reported_at=datetime.now() - timedelta(days=7),
        platform="kakao"
    ),
    ScamCase(
        id="case_006",
        title="해외 사업가 사칭 스캠",
        description="해외 무역업자 사칭, 사업 자금 문제로 3500만원 편취",
        phrases_used=["id_003", "lb_002", "fr_001", "ur_003", "is_002"],
        damage_amount=35000000,
        reported_at=datetime.now() - timedelta(days=90),
        platform="instagram"
    ),
    ScamCase(
        id="case_007",
        title="결혼 약속 후 금전 요구",
        description="결혼 약속하며 한국 입국 비용 명목으로 2500만원 요구",
        phrases_used=["ff_001", "ff_002", "lb_005", "fr_002", "av_002"],
        damage_amount=25000000,
        reported_at=datetime.now() - timedelta(days=20),
        platform="kakao"
    ),
    ScamCase(
        id="case_008",
        title="코인 투자 사기",
        description="연인처럼 행동하며 코인 투자 플랫폼 가입 유도 후 1억원 피해",
        phrases_used=["lb_001", "fr_004", "fr_005", "ur_004", "is_001"],
        damage_amount=100000000,
        reported_at=datetime.now() - timedelta(days=10),
        platform="telegram"
    ),
    ScamCase(
        id="case_009",
        title="의사 사칭 로맨스 스캠",
        description="UN 소속 의사 사칭, 의료장비 구입비 명목으로 4000만원 편취",
        phrases_used=["id_002", "lb_002", "av_003", "fr_002", "ur_001"],
        damage_amount=40000000,
        reported_at=datetime.now() - timedelta(days=55),
        platform="instagram"
    ),
    ScamCase(
        id="case_010",
        title="고립 유도 후 금전 요구",
        description="가족에게 비밀로 하라며 고립시킨 후 반복적 금전 요구, 6000만원 피해",
        phrases_used=["is_001", "is_002", "lb_001", "fr_003", "ss_003"],
        damage_amount=60000000,
        reported_at=datetime.now() - timedelta(days=40),
        platform="kakao"
    ),
]


async def seed_data():
    """샘플 데이터 시드"""
    repo = ScamPatternRepository()
    await repo.initialize()

    if not repo.is_connected():
        print("Neo4j 연결 실패")
        return

    print("스캠 문구 저장 중...")
    for phrase in SCAM_PHRASES:
        await repo.save_phrase(phrase)
        print(f"  - {phrase.id}: {phrase.text}")

    print("\n스캠 사례 저장 중...")
    for case in SCAM_CASES:
        await repo.save_case(case)
        print(f"  - {case.id}: {case.title}")

    stats = await repo.get_stats()
    print(f"\n시드 완료: {stats['phrases']}개 문구, {stats['cases']}개 사례")

    await repo.close()


if __name__ == "__main__":
    asyncio.run(seed_data())
