from dataclasses import dataclass


@dataclass(frozen=True)
class Platform:
    """플랫폼 값 객체"""
    name: str
    icon: str
    color: str
    base_url: str

    @classmethod
    def instagram(cls) -> "Platform":
        return cls("Instagram", "📷", "#E4405F", "https://instagram.com")

    @classmethod
    def facebook(cls) -> "Platform":
        return cls("Facebook", "👤", "#1877F2", "https://facebook.com")

    @classmethod
    def twitter(cls) -> "Platform":
        return cls("X", "✕", "#000000", "https://x.com")

    @classmethod
    def linkedin(cls) -> "Platform":
        return cls("LinkedIn", "💼", "#0A66C2", "https://linkedin.com/in")

    @classmethod
    def google(cls) -> "Platform":
        return cls("Google", "🔍", "#4285F4", "https://images.google.com")


@dataclass(frozen=True)
class ProfileMatch:
    """프로필 매칭 결과 값 객체"""
    platform: str
    name: str
    username: str
    profile_url: str
    image_url: str
    match_score: int


@dataclass(frozen=True)
class ScammerMatch:
    """스캐머 매칭 결과 값 객체"""
    scammer_id: str
    name: str
    confidence: int
    report_count: int
    distance: float


@dataclass(frozen=True)
class ReverseSearchLink:
    """역이미지 검색 링크 값 객체"""
    platform: str
    name: str
    url: str
    icon: str

    @classmethod
    def google(cls) -> "ReverseSearchLink":
        return cls("google", "Google 이미지", "https://images.google.com/", "🔍")

    @classmethod
    def yandex(cls) -> "ReverseSearchLink":
        return cls("yandex", "Yandex (얼굴 검색 강력)", "https://yandex.com/images/", "🔎")

    @classmethod
    def tineye(cls) -> "ReverseSearchLink":
        return cls("tineye", "TinEye", "https://tineye.com/", "👁️")

    @classmethod
    def bing(cls) -> "ReverseSearchLink":
        return cls("bing", "Bing 이미지", "https://www.bing.com/visualsearch", "🔷")

    @classmethod
    def all_links(cls) -> list["ReverseSearchLink"]:
        return [cls.google(), cls.yandex(), cls.tineye(), cls.bing()]
