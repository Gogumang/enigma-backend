import logging
from dataclasses import dataclass, field

from src.domain.profile import ProfileMatch, ReverseSearchLink, ScammerMatch
from src.domain.scammer import ScammerEntity, ScammerRepository
from src.infrastructure.ai import FaceRecognitionService
from src.infrastructure.external import ImageSearchScraper, SocialMediaSearcher
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


@dataclass
class ProfileSearchResult:
    """프로필 검색 결과 DTO"""
    total_found: int
    scammer_matches: list[ScammerMatch]
    reverse_search_links: list[ReverseSearchLink]
    profile_matches: dict[str, list[ProfileMatch]]
    web_image_results: list[WebImageResult] = field(default_factory=list)
    uploaded_image_url: str | None = None


@dataclass
class ReportScammerResult:
    """스캐머 신고 결과 DTO"""
    success: bool
    message: str
    scammer_id: str | None = None


class ProfileSearchUseCase:
    """프로필 검색 유스케이스"""

    def __init__(
        self,
        face_recognition: FaceRecognitionService,
        scammer_repository: ScammerRepository,
        image_search_scraper: ImageSearchScraper | None = None,
        social_media_searcher: SocialMediaSearcher | None = None
    ):
        self.face_recognition = face_recognition
        self.scammer_repository = scammer_repository
        self.image_search_scraper = image_search_scraper or ImageSearchScraper(face_recognition)
        self.social_media_searcher = social_media_searcher or SocialMediaSearcher()

    async def search_by_image(
        self,
        image_data: bytes,
        query: str | None = None
    ) -> ProfileSearchResult:
        """이미지로 프로필 검색"""
        scammer_matches = []
        web_image_results = []
        uploaded_image_url = None

        # 1. 얼굴 임베딩 추출 및 스캐머 DB 비교
        embedding = await self.face_recognition.extract_embedding(image_data)

        if embedding:
            matches = await self.scammer_repository.find_by_face_embedding(
                embedding, threshold=0.7
            )

            if matches:
                for scammer, distance in matches:
                    confidence = max(0, int((1 - distance) * 100))
                    scammer_matches.append(ScammerMatch(
                        scammer_id=scammer.id,
                        name=scammer.name,
                        confidence=confidence,
                        report_count=scammer.report_count,
                        distance=distance
                    ))
                logger.info(f"스캐머 DB에서 {len(matches)}개 일치 발견")
            else:
                logger.info("스캐머 DB에서 일치하는 얼굴 없음")
        else:
            logger.info("이미지에서 얼굴을 감지하지 못함")

        # 2. 역이미지 검색 실행 (Yandex, Google 등에서 실제 결과 가져오기)
        reverse_links = []
        logger.info("역이미지 검색 시작 (웹 스크래핑)...")

        try:
            search_result = await self.image_search_scraper.search(image_data)

            if search_result.success:
                uploaded_image_url = search_result.uploaded_image_url

                for result in search_result.results:
                    web_image_results.append(WebImageResult(
                        title=result.title,
                        source_url=result.source_url,
                        image_url=result.image_url,
                        thumbnail_url=result.thumbnail_url,
                        platform=result.platform,
                        match_score=result.match_score
                    ))

                logger.info(f"역이미지 검색 완료: {len(web_image_results)}개 결과")
            else:
                logger.warning(f"역이미지 검색 실패: {search_result.error}")
                reverse_links = ReverseSearchLink.all_links()

        except Exception as e:
            logger.error(f"역이미지 검색 중 오류: {e}")
            reverse_links = ReverseSearchLink.all_links()

        # 3. 이름이 제공된 경우 소셜 미디어 검색
        profile_matches: dict[str, list[ProfileMatch]] = {
            "instagram": [],
            "facebook": [],
            "twitter": [],
            "linkedin": [],
            "google": []
        }

        if query and query.strip():
            # 소셜 미디어 프로필 검색
            try:
                social_results = await self.social_media_searcher.search_by_name(query.strip())
                for result in social_results:
                    web_image_results.append(WebImageResult(
                        title=result.title,
                        source_url=result.source_url,
                        image_url=result.image_url,
                        thumbnail_url=result.thumbnail_url,
                        platform=result.platform,
                        match_score=result.match_score
                    ))
            except Exception as e:
                logger.warning(f"소셜 미디어 검색 실패: {e}")

            # 플랫폼별 검색 링크도 생성
            profile_matches = self._generate_search_links(query.strip())

        total_found = len(scammer_matches) + len(web_image_results)

        return ProfileSearchResult(
            total_found=total_found,
            scammer_matches=scammer_matches,
            reverse_search_links=reverse_links,
            profile_matches=profile_matches,
            web_image_results=web_image_results,
            uploaded_image_url=uploaded_image_url
        )

    async def search_by_query(self, query: str) -> ProfileSearchResult:
        """텍스트로 프로필 검색"""
        if not query.strip():
            raise ValidationException("검색어를 입력해주세요")

        profile_matches = self._generate_search_links(query.strip())

        return ProfileSearchResult(
            total_found=0,
            scammer_matches=[],
            reverse_search_links=[],
            profile_matches=profile_matches
        )

    def _generate_search_links(self, query: str) -> dict[str, list[ProfileMatch]]:
        """플랫폼별 검색 링크 생성 (실제 검색 페이지로 이동)"""
        from urllib.parse import quote

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


class ReportScammerUseCase:
    """스캐머 신고 유스케이스"""

    def __init__(
        self,
        face_recognition: FaceRecognitionService,
        scammer_repository: ScammerRepository
    ):
        self.face_recognition = face_recognition
        self.scammer_repository = scammer_repository

    async def report(
        self,
        image_data: bytes,
        name: str,
        source: str | None = None
    ) -> ReportScammerResult:
        """스캐머 신고"""
        if not name.strip():
            raise ValidationException("스캐머 이름을 입력해주세요")

        # 1. 얼굴 임베딩 추출
        embedding = await self.face_recognition.extract_embedding(image_data)

        if not embedding:
            return ReportScammerResult(
                success=False,
                message="이미지에서 얼굴을 감지할 수 없습니다"
            )

        # 2. 기존 스캐머 확인
        existing = await self.scammer_repository.find_by_face_embedding(
            embedding, threshold=0.4
        )

        if existing:
            # 이미 등록된 스캐머 - 신고 횟수 증가
            scammer, distance = existing[0]
            scammer.increment_report()
            await self.scammer_repository.save(scammer)

            return ReportScammerResult(
                success=True,
                message=f"이미 등록된 스캐머입니다. 신고 횟수: {scammer.report_count}",
                scammer_id=scammer.id
            )

        # 3. 새 스캐머 등록
        new_scammer = ScammerEntity.create(
            name=name.strip(),
            face_embedding=embedding,
            source=source
        )

        await self.scammer_repository.save(new_scammer)

        logger.info(f"New scammer registered: {new_scammer.id}")

        return ReportScammerResult(
            success=True,
            message="스캐머가 데이터베이스에 등록되었습니다",
            scammer_id=new_scammer.id
        )
