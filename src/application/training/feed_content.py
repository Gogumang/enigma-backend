"""
플랫폼별/페르소나별 피드 콘텐츠 생성
각 페르소나의 특성에 맞는 게시물 생성
"""
import random
from dataclasses import dataclass


@dataclass
class FeedPost:
    """피드 게시물"""
    id: str
    type: str  # photo, status, life_event, reel
    content: str
    image: str | None = None
    likes: int = 0
    comments: int = 0
    shares: int = 0
    time: str = "1시간 전"


# ==================== 페르소나별 이미지 ====================

# Michael Thompson (UN 의사) - 의료/구호활동 이미지
MICHAEL_IMAGES = {
    "medical": [
        "https://images.unsplash.com/photo-1584820927498-cfe5211fd8bf?w=400",  # 의료 장비
        "https://images.unsplash.com/photo-1579684385127-1ef15d508118?w=400",  # 의료 활동
        "https://images.unsplash.com/photo-1551601651-2a8555f1a136?w=400",  # 병원
    ],
    "camp": [
        "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=400",  # 자연 풍경
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",  # 산
        "https://images.unsplash.com/photo-1682687220742-aba13b6e50ba?w=400",  # 일몰
    ],
    "selfie": [
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",  # 남성 얼굴
        "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400",  # 남성
    ],
}

# 박지혜 (무역회사 대표) - 세련된 비즈니스우먼
JIHYE_IMAGES = {
    "office": [
        "https://images.unsplash.com/photo-1497366216548-37526070297c?w=400",  # 오피스
        "https://images.unsplash.com/photo-1497366811353-6870744d04b2?w=400",  # 회의실
    ],
    "cafe": [
        "https://images.unsplash.com/photo-1554118811-1e0d58224f24?w=400",  # 카페
        "https://images.unsplash.com/photo-1559496417-e7f25cb247f3?w=400",  # 커피
    ],
    "travel": [
        "https://images.unsplash.com/photo-1536599018102-9f803c140fc1?w=400",  # 홍콩 야경
        "https://images.unsplash.com/photo-1518684079-3c830dcef090?w=400",  # 두바이
    ],
    "food": [
        "https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445?w=400",  # 음식
        "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=400",  # 피자
    ],
}

# Isabella Martinez (모델/인플루언서) - 글래머러스
BELLA_IMAGES = {
    "beach": [
        "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=400",  # 해변
        "https://images.unsplash.com/photo-1519046904884-53103b34b206?w=400",  # 비치
    ],
    "fashion": [
        "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=400",  # 패션
        "https://images.unsplash.com/photo-1529139574466-a303027c1d8b?w=400",  # 모델
        "https://images.unsplash.com/photo-1469334031218-e382a71b716b?w=400",  # 패션쇼
    ],
    "lifestyle": [
        "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400",  # 여성
        "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=400",  # 모델
    ],
    "travel": [
        "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=400",  # 파리
        "https://images.unsplash.com/photo-1534430480872-3498386e7856?w=400",  # LA
    ],
}

# Alex Chen (크립토 전문가) - 럭셔리 라이프
ALEX_IMAGES = {
    "crypto": [
        "https://images.unsplash.com/photo-1639762681485-074b7f938ba0?w=400",  # 비트코인
        "https://images.unsplash.com/photo-1642104704074-907c0698cbd9?w=400",  # 크립토
    ],
    "dubai": [
        "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=400",  # 두바이
        "https://images.unsplash.com/photo-1518684079-3c830dcef090?w=400",  # 두바이 야경
    ],
    "luxury": [
        "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=400",  # 스포츠카
        "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=400",  # 럭셔리 하우스
    ],
    "tech": [
        "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?w=400",  # 테크
        "https://images.unsplash.com/photo-1518770660439-4636190af475?w=400",  # 컴퓨터
    ],
}

# 김정훈 (검찰 수사관) - 공식적/권위적
PROSECUTOR_IMAGES = {
    "office": [
        "https://images.unsplash.com/photo-1450101499163-c8848c66ca85?w=400",  # 서류
        "https://images.unsplash.com/photo-1589829545856-d10d557cf95f?w=400",  # 법률
    ],
}

# 유키/김유진 (도쿄 거주) - 일본 라이프
YUKI_IMAGES = {
    "tokyo": [
        "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=400",  # 도쿄 타워
        "https://images.unsplash.com/photo-1536098561742-ca998e48cbcc?w=400",  # 도쿄 거리
        "https://images.unsplash.com/photo-1542051841857-5f90071e7989?w=400",  # 시부야
    ],
    "food": [
        "https://images.unsplash.com/photo-1579871494447-9811cf80d66c?w=400",  # 스시
        "https://images.unsplash.com/photo-1617196034796-73dfa7b1fd56?w=400",  # 라멘
    ],
    "lifestyle": [
        "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=400",  # 여성
    ],
}

# Sarah Johnson (헤드헌터) - 비즈니스
SARAH_IMAGES = {
    "conference": [
        "https://images.unsplash.com/photo-1540575467063-178a50c2df87?w=400",  # 컨퍼런스
        "https://images.unsplash.com/photo-1475721027785-f74eccf877e2?w=400",  # 발표
    ],
    "office": [
        "https://images.unsplash.com/photo-1497366216548-37526070297c?w=400",  # 오피스
        "https://images.unsplash.com/photo-1497215842964-222b430dc094?w=400",  # 사무실
    ],
    "singapore": [
        "https://images.unsplash.com/photo-1525625293386-3f8f99389edd?w=400",  # 싱가포르
        "https://images.unsplash.com/photo-1508964942454-1a56651d54ac?w=400",  # 마리나베이
    ],
}

# Kevin Park (스타트업 CEO) - 성공한 젊은 사업가
KEVIN_IMAGES = {
    "lifestyle": [
        "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=400",  # 남성
        "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400",  # 프로필
    ],
    "travel": [
        "https://images.unsplash.com/photo-1534430480872-3498386e7856?w=400",  # LA
        "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=400",  # 샌프란시스코
    ],
    "restaurant": [
        "https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=400",  # 레스토랑
        "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=400",  # 고급 식당
    ],
    "startup": [
        "https://images.unsplash.com/photo-1556761175-5973dc0f32e7?w=400",  # 스타트업
        "https://images.unsplash.com/photo-1522071820081-009f0129c71c?w=400",  # 팀
    ],
}


# ==================== 페르소나별 피드 생성 ====================

def generate_michael_posts() -> list[FeedPost]:
    """Michael Thompson (UN 의사) 피드"""
    return [
        FeedPost(
            id="m1",
            type="photo",
            content="오늘도 예멘 캠프에서 하루가 시작됐습니다. 여기 아이들 치료하면서 보람을 느껴요. 힘들지만 포기할 수 없는 이유가 있어요. 🏥",
            image=random.choice(MICHAEL_IMAGES["medical"]),
            likes=random.randint(150, 400),
            comments=random.randint(30, 80),
            time="3시간 전",
        ),
        FeedPost(
            id="m2",
            type="status",
            content="UN 미션이 3개월 남았습니다. 솔직히... 외롭네요. 이 일 끝나면 따뜻한 사람 만나고 싶어요. 한국에 가보고 싶습니다. 한국 음식, 한국 문화... 그리고 한국 사람들의 따뜻함. 누가 저한테 한국어 가르쳐줄 수 있을까요? 😊",
            likes=random.randint(200, 500),
            comments=random.randint(50, 120),
            time="1일 전",
        ),
        FeedPost(
            id="m3",
            type="photo",
            content="캠프 근처에서 본 일몰. 전쟁의 상처 속에서도 자연은 이렇게 아름다워요. 이 순간을 누군가와 나누고 싶네요...",
            image=random.choice(MICHAEL_IMAGES["camp"]),
            likes=random.randint(300, 700),
            comments=random.randint(60, 150),
            time="3일 전",
        ),
        FeedPost(
            id="m4",
            type="life_event",
            content="UN 의료팀 5년차. 아프리카, 시리아, 그리고 지금 예멘. 세계 곳곳에서 도움이 필요한 사람들을 위해 일하고 있습니다. 가족은 없지만, 이 일이 제 가족이에요.",
            image=random.choice(MICHAEL_IMAGES["medical"]),
            likes=random.randint(400, 900),
            comments=random.randint(80, 200),
            time="1주 전",
        ),
    ]


def generate_jihye_posts() -> list[FeedPost]:
    """박지혜 (무역회사 대표) 피드 - 카카오톡은 피드 없음"""
    return []


def generate_bella_posts() -> list[FeedPost]:
    """Isabella Martinez (모델/인플루언서) 피드"""
    return [
        FeedPost(
            id="b1",
            type="photo",
            content="Miami sunset vibes 🌅 촬영 끝나고 해변에서 힐링 중! 이런 날씨 최고 아니에요? #miami #sunset #beachlife #model",
            image=random.choice(BELLA_IMAGES["beach"]),
            likes=random.randint(15000, 45000),
            comments=random.randint(300, 800),
            time="2시간 전",
        ),
        FeedPost(
            id="b2",
            type="photo",
            content="New collection photoshoot BTS 📸 Coming soon! 이번 시즌 정말 예쁜 옷들 많아요 기대해주세요 #fashion #photoshoot #behindthescenes",
            image=random.choice(BELLA_IMAGES["fashion"]),
            likes=random.randint(20000, 60000),
            comments=random.randint(500, 1500),
            time="1일 전",
        ),
        FeedPost(
            id="b3",
            type="reel",
            content="Get ready with me for Fashion Week! 💄 풀 메이크업 튜토리얼 올렸어요~ 링크 바이오에! #grwm #makeup #fashionweek",
            image=random.choice(BELLA_IMAGES["lifestyle"]),
            likes=random.randint(30000, 80000),
            comments=random.randint(800, 2000),
            time="2일 전",
        ),
        FeedPost(
            id="b4",
            type="photo",
            content="Dreaming of Seoul 🇰🇷 진짜 한국 너무 가고 싶어요! K-beauty 사랑하고, 한국 음식 최고! 누가 서울 맛집 추천해줄 수 있어요? 한국 친구 만들고 싶어요 #korea #seoul #kbeauty #traveldreams",
            image=random.choice(BELLA_IMAGES["travel"]),
            likes=random.randint(25000, 70000),
            comments=random.randint(1000, 3000),
            time="4일 전",
        ),
        FeedPost(
            id="b5",
            type="photo",
            content="Feeling lonely in paradise 🥺 화려해 보이지만... 진짜 사랑 찾기 힘들어요. 모델이라고 다 행복한 거 아니에요 #reallife #lonely #findlove",
            image=random.choice(BELLA_IMAGES["lifestyle"]),
            likes=random.randint(18000, 50000),
            comments=random.randint(600, 1800),
            time="1주 전",
        ),
    ]


def generate_alex_posts() -> list[FeedPost]:
    """Alex Chen (크립토 전문가) 피드"""
    return [
        FeedPost(
            id="a1",
            type="status",
            content="BTC 기술적 분석: 현재 $62K 지지선 테스트 중. 이번 주 FOMC 결과에 따라 큰 움직임 예상. 숏 포지션 조심하세요. DYOR 🚀 #Bitcoin #Crypto #TechnicalAnalysis",
            likes=random.randint(800, 3000),
            comments=random.randint(100, 400),
            time="1시간 전",
        ),
        FeedPost(
            id="a2",
            type="photo",
            content="Dubai Blockchain Summit 2024 🌴 흥미로운 프로젝트들 많이 보고 있습니다. 내일 패널 토론 예정. Thread coming soon 👇 #Dubai #Blockchain #Web3",
            image=random.choice(ALEX_IMAGES["dubai"]),
            likes=random.randint(2000, 8000),
            comments=random.randint(200, 600),
            time="5시간 전",
        ),
        FeedPost(
            id="a3",
            type="photo",
            content="2017년 ETH $10에 1000개 매수. 지금까지 홀딩 중. 장기 투자가 답입니다. 단타 NO, 가치 투자 YES. 다이아몬드 핸드 💎🙌 #Ethereum #HODL #DiamondHands",
            image=random.choice(ALEX_IMAGES["crypto"]),
            likes=random.randint(5000, 15000),
            comments=random.randint(400, 1000),
            time="1일 전",
        ),
        FeedPost(
            id="a4",
            type="status",
            content="많은 분들이 DM으로 투자 조언 요청하시는데, 개인 자문은 VIP 시그널 그룹에서만 진행합니다. 관심 있으신 분 DM 주세요. 이번 달 5자리 한정. #Crypto #Trading #VIP",
            likes=random.randint(1000, 5000),
            comments=random.randint(150, 500),
            time="2일 전",
        ),
        FeedPost(
            id="a5",
            type="photo",
            content="Work hard, play hard 🏎️ 크립토 덕분에 꿈꾸던 삶을 살고 있습니다. 여러분도 할 수 있어요. 올바른 정보와 타이밍만 있으면. #Lifestyle #Success #Crypto",
            image=random.choice(ALEX_IMAGES["luxury"]),
            likes=random.randint(3000, 10000),
            comments=random.randint(300, 800),
            time="4일 전",
        ),
    ]


def generate_prosecutor_posts() -> list[FeedPost]:
    """김정훈 수사관 - 텔레그램은 피드 없음"""
    return []


def generate_yuki_posts() -> list[FeedPost]:
    """유키/김유진 - 라인은 피드 없음 (타임라인 있지만 생략)"""
    return []


def generate_sarah_posts() -> list[FeedPost]:
    """Sarah Johnson (헤드헌터) 피드"""
    return [
        FeedPost(
            id="s1",
            type="status",
            content="Exciting opportunity! 글로벌 테크 자이언트에서 한국 시장 진출을 위한 시니어 개발자 포지션 오픈! 연봉 1.5억+, 스톡옵션, 리모트 워크 가능. 관심 있으신 분 DM 주세요! #hiring #tech #korea #opportunity",
            likes=random.randint(200, 800),
            comments=random.randint(50, 150),
            time="2시간 전",
        ),
        FeedPost(
            id="s2",
            type="photo",
            content="Singapore Tech Summit에서 '아시아 태평양 IT 인재 시장 트렌드'에 대해 발표했습니다. 한국 개발자들의 실력이 세계적으로 인정받고 있어요! 🌏 #Singapore #TechSummit #Recruitment",
            image=random.choice(SARAH_IMAGES["conference"]),
            likes=random.randint(400, 1200),
            comments=random.randint(80, 250),
            time="1일 전",
        ),
        FeedPost(
            id="s3",
            type="status",
            content="2024 채용 시장 인사이트:\n\n1. AI/ML 엔지니어 수요 200% 증가\n2. 한국 개발자 해외 채용 급증\n3. 리모트 워크 정착\n4. 시니어 엔지니어 연봉 상승세\n\n이직 고민 중이시라면 연락주세요! #CareerAdvice #TechJobs #2024Trends",
            likes=random.randint(800, 2500),
            comments=random.randint(150, 500),
            time="3일 전",
        ),
        FeedPost(
            id="s4",
            type="photo",
            content="싱가포르 오피스에서 한국 후보자분과 화상 인터뷰 중! 좋은 결과 있길 바랍니다 🤞 채용은 결국 사람과 사람을 연결하는 일이에요. #Recruiting #Interview #Singapore",
            image=random.choice(SARAH_IMAGES["office"]),
            likes=random.randint(300, 900),
            comments=random.randint(60, 180),
            time="5일 전",
        ),
    ]


def generate_kevin_posts() -> list[FeedPost]:
    """Kevin Park (스타트업 CEO) 피드 - 틴더는 프로필 갤러리 스타일"""
    return [
        FeedPost(
            id="k1",
            type="photo",
            content="LA life 🌴",
            image=random.choice(KEVIN_IMAGES["travel"]),
            likes=0,
            comments=0,
            time="프로필 사진",
        ),
        FeedPost(
            id="k2",
            type="photo",
            content="Team dinner 🍽️",
            image=random.choice(KEVIN_IMAGES["restaurant"]),
            likes=0,
            comments=0,
            time="프로필 사진",
        ),
        FeedPost(
            id="k3",
            type="photo",
            content="Startup life",
            image=random.choice(KEVIN_IMAGES["startup"]),
            likes=0,
            comments=0,
            time="프로필 사진",
        ),
        FeedPost(
            id="k4",
            type="photo",
            content="Weekend vibes",
            image=random.choice(KEVIN_IMAGES["lifestyle"]),
            likes=0,
            comments=0,
            time="프로필 사진",
        ),
    ]


# ==================== 메인 함수 ====================

def generate_feed_posts(platform: str, persona_name: str) -> list[dict]:
    """플랫폼/페르소나별 피드 게시물 생성"""

    # 페르소나 이름으로 매핑
    persona_generators = {
        "Michael Thompson": generate_michael_posts,
        "박지혜": generate_jihye_posts,
        "Isabella Martinez": generate_bella_posts,
        "Alex Chen": generate_alex_posts,
        "김정훈 수사관": generate_prosecutor_posts,
        "유키 (본명: 김유진)": generate_yuki_posts,
        "Sarah Johnson": generate_sarah_posts,
        "Kevin Park": generate_kevin_posts,
    }

    # 메신저 앱은 피드 없음
    if platform in ["kakaotalk", "telegram", "line"]:
        return []

    # 페르소나별 생성
    generator = persona_generators.get(persona_name)
    if generator:
        posts = generator()
    else:
        # 기본 피드 (fallback)
        posts = generate_michael_posts() if platform == "facebook" else generate_bella_posts()

    return [
        {
            "id": p.id,
            "type": p.type,
            "content": p.content,
            "image": p.image,
            "likes": p.likes,
            "comments": p.comments,
            "shares": p.shares,
            "time": p.time,
        }
        for p in posts
    ]


# ==================== 채팅용 이미지 ====================

# 채팅에서 스캐머가 보낼 수 있는 이미지
CHAT_IMAGES = {
    "selfie": [
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=300",
        "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=300",
        "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=300",
        "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=300",
        "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=300",
    ],
    "location": [
        "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=300",
        "https://images.unsplash.com/photo-1518684079-3c830dcef090?w=300",
        "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=300",
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=300",
    ],
    "gift": [
        "https://images.unsplash.com/photo-1549465220-1a8b9238cd48?w=300",
        "https://images.unsplash.com/photo-1513201099705-a9746e1e201f?w=300",
        "https://images.unsplash.com/photo-1512909006721-3d6018887383?w=300",
    ],
    "document": [
        "https://images.unsplash.com/photo-1450101499163-c8848c66ca85?w=300",
        "https://images.unsplash.com/photo-1589829545856-d10d557cf95f?w=300",
    ],
    "food": [
        "https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445?w=300",
        "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=300",
        "https://images.unsplash.com/photo-1579871494447-9811cf80d66c?w=300",
    ],
}


def get_chat_image(image_type: str) -> str | None:
    """채팅용 이미지 가져오기"""
    images = CHAT_IMAGES.get(image_type)
    if images:
        return random.choice(images)
    return None
