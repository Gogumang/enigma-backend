import logging
from dataclasses import dataclass, field
from urllib.parse import quote

from src.domain.profile import ProfileMatch, ReverseSearchLink, ScammerMatch
from src.infrastructure.ai import FaceRecognitionService
from src.infrastructure.external import (
    EnhancedImageSearchService,
    EnhancedSocialMediaSearcher,
    ImageSearchScraper,
    SocialMediaSearcher,
)
from src.shared.exceptions import ValidationException

logger = logging.getLogger(__name__)


@dataclass
class WebImageResult:
    """웹에서 발견된 이미지"""
    title: str
    source_url: str
    image_url: str
    thumbnail_url: str | None
    platform: str  # instagram, facebook, linkedin, twitter, other
    match_score: float  # 0-100
    source_engine: str = "unknown"  # 어떤 검색 엔진에서 왔는지


@dataclass
class SearchLink:
    """검색 링크"""
    name: str
    url: str
    icon: str
    category: str  # reverse_image, face_search, scam_check, social_media
    description: str
    priority: int = 0


@dataclass
class ProfileSearchResult:
    """프로필 검색 결과 DTO"""
    total_found: int
    scammer_matches: list[ScammerMatch]
    reverse_search_links: list[ReverseSearchLink]
    profile_matches: dict[str, list[ProfileMatch]]
    web_image_results: list[WebImageResult] = field(default_factory=list)
    search_links: list[SearchLink] = field(default_factory=list)
    social_search_links: list[dict] = field(default_factory=list)
    uploaded_image_url: str | None = None


class ProfileSearchUseCase:
    """프로필 검색 유스케이스 - 강화된 버전"""

    def __init__(
        self,
        face_recognition: FaceRecognitionService,
        image_search_scraper: ImageSearchScraper | None = None,
        social_media_searcher: SocialMediaSearcher | None = None
    ):
        self.face_recognition = face_recognition
        # 기존 서비스 (하위 호환성)
        self.image_search_scraper = image_search_scraper or ImageSearchScraper(face_recognition)
        self.social_media_searcher = social_media_searcher or SocialMediaSearcher()
        # 강화된 서비스
        self.enhanced_search = EnhancedImageSearchService(face_recognition)
        self.enhanced_social = EnhancedSocialMediaSearcher()

    async def search_by_image(
        self,
        image_data: bytes,
        query: str | None = None
    ) -> ProfileSearchResult:
        """이미지로 프로필 검색 (강화된 버전)"""
        web_image_results = []
        search_links = []
        social_search_links = []
        uploaded_image_url = None

        # 1. 강화된 역이미지 검색 실행
        logger.info("강화된 역이미지 검색 시작...")

        try:
            enhanced_result = await self.enhanced_search.search(image_data)

            if enhanced_result.success:
                uploaded_image_url = enhanced_result.uploaded_image_url

                # 검색 결과 변환
                for result in enhanced_result.results:
                    web_image_results.append(WebImageResult(
                        title=result.title,
                        source_url=result.source_url,
                        image_url=result.image_url,
                        thumbnail_url=result.thumbnail_url,
                        platform=result.platform,
                        match_score=result.match_score,
                        source_engine=result.source_engine
                    ))

                # 검색 링크 변환
                for link in enhanced_result.search_links:
                    search_links.append(SearchLink(
                        name=link["name"],
                        url=link["url"],
                        icon=link.get("icon", ""),
                        category=link.get("category", "other"),
                        description=link.get("description", ""),
                        priority=link.get("priority", 0)
                    ))

                logger.info(f"강화된 검색 완료: {len(web_image_results)}개 결과")
            else:
                logger.warning(f"강화된 검색 실패: {enhanced_result.error}")
                # 폴백: 기존 검색 사용
                await self._fallback_search(image_data, web_image_results)

        except Exception as e:
            logger.error(f"강화된 검색 중 오류: {e}")
            await self._fallback_search(image_data, web_image_results)

        # 3. 이름/쿼리로 소셜 미디어 검색 링크 생성
        profile_matches: dict[str, list[ProfileMatch]] = {
            "instagram": [],
            "facebook": [],
            "twitter": [],
            "linkedin": [],
            "google": []
        }

        if query and query.strip():
            # 강화된 소셜 미디어 검색 링크
            social_search_links = self.enhanced_social.generate_search_links(query.strip())

            # 기존 프로필 매치 형식도 유지 (하위 호환성)
            profile_matches = self._generate_search_links(query.strip())

        # 검색 링크가 없으면 기본 링크 추가
        if not search_links:
            search_links = [
                SearchLink(
                    name=link.name,
                    url=link.url,
                    icon="",
                    category="reverse_image",
                    description=link.name,
                    priority=0
                )
                for link in ReverseSearchLink.all_links()
            ]

        total_found = len(web_image_results)

        return ProfileSearchResult(
            total_found=total_found,
            scammer_matches=[],
            reverse_search_links=[],  # 이제 search_links로 대체
            profile_matches=profile_matches,
            web_image_results=web_image_results,
            search_links=search_links,
            social_search_links=social_search_links,
            uploaded_image_url=uploaded_image_url
        )

    async def _fallback_search(
        self,
        image_data: bytes,
        web_image_results: list[WebImageResult]
    ) -> None:
        """기존 검색으로 폴백"""
        try:
            search_result = await self.image_search_scraper.search(image_data)

            if search_result.success:
                for result in search_result.results:
                    web_image_results.append(WebImageResult(
                        title=result.title,
                        source_url=result.source_url,
                        image_url=result.image_url,
                        thumbnail_url=result.thumbnail_url,
                        platform=result.platform,
                        match_score=result.match_score,
                        source_engine="fallback"
                    ))
        except Exception as e:
            logger.error(f"폴백 검색 실패: {e}")

    async def search_by_query(self, query: str) -> ProfileSearchResult:
        """텍스트로 프로필 검색"""
        if not query.strip():
            raise ValidationException("검색어를 입력해주세요")

        profile_matches = self._generate_search_links(query.strip())
        social_search_links = self.enhanced_social.generate_search_links(query.strip())

        return ProfileSearchResult(
            total_found=0,
            scammer_matches=[],
            reverse_search_links=[],
            profile_matches=profile_matches,
            social_search_links=social_search_links
        )

    def _generate_search_links(self, query: str) -> dict[str, list[ProfileMatch]]:
        """플랫폼별 검색 링크 생성 (하위 호환성)"""
        encoded_query = quote(query)

        return {
            "instagram": [
                ProfileMatch(
                    platform="instagram",
                    name=f"'{query}' 검색하기",
                    username="instagram.com",
                    profile_url=f"https://www.instagram.com/explore/search/keyword/?q={encoded_query}",
                    image_url="https://ui-avatars.com/api/?name=IG&background=E4405F&color=fff",
                    match_score=0
                )
            ],
            "facebook": [
                ProfileMatch(
                    platform="facebook",
                    name=f"'{query}' 검색하기",
                    username="facebook.com",
                    profile_url=f"https://www.facebook.com/search/people/?q={encoded_query}",
                    image_url="https://ui-avatars.com/api/?name=FB&background=1877F2&color=fff",
                    match_score=0
                )
            ],
            "twitter": [
                ProfileMatch(
                    platform="twitter",
                    name=f"'{query}' 검색하기",
                    username="x.com",
                    profile_url=f"https://x.com/search?q={encoded_query}&f=user",
                    image_url="https://ui-avatars.com/api/?name=X&background=000000&color=fff",
                    match_score=0
                )
            ],
            "linkedin": [
                ProfileMatch(
                    platform="linkedin",
                    name=f"'{query}' 검색하기",
                    username="linkedin.com",
                    profile_url=f"https://www.linkedin.com/search/results/people/?keywords={encoded_query}",
                    image_url="https://ui-avatars.com/api/?name=IN&background=0A66C2&color=fff",
                    match_score=0
                )
            ],
            "google": [
                ProfileMatch(
                    platform="google",
                    name=f"'{query}' 검색하기",
                    username="google.com",
                    profile_url=f"https://www.google.com/search?q={encoded_query}",
                    image_url="https://ui-avatars.com/api/?name=G&background=4285F4&color=fff",
                    match_score=0
                )
            ]
        }


