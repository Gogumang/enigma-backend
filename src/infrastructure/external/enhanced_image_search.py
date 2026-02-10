"""
강화된 역이미지 검색 서비스
- SerpApi (Google Lens) 통합
- 다양한 얼굴 검색 서비스 링크 제공
- 소셜 미디어 검색 강화
"""
import asyncio
import base64
import logging
from dataclasses import dataclass
from urllib.parse import quote

import httpx

from src.shared.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSearchResult:
    """강화된 검색 결과"""
    title: str
    source_url: str
    image_url: str
    thumbnail_url: str | None
    platform: str
    match_score: float
    source_engine: str  # 어떤 검색 엔진에서 왔는지


@dataclass
class EnhancedSearchResponse:
    """강화된 검색 응답"""
    success: bool
    results: list[EnhancedSearchResult]
    search_links: list[dict]  # 수동 검색 링크
    uploaded_image_url: str | None = None
    error: str | None = None


class EnhancedImageSearchService:
    """강화된 역이미지 검색 서비스"""

    def __init__(self, face_recognition_service=None):
        self.settings = get_settings()
        self.face_recognition = face_recognition_service
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
        }

    async def search(self, image_data: bytes) -> EnhancedSearchResponse:
        """이미지로 강화된 역이미지 검색 수행"""
        results = []
        search_links = []
        uploaded_image_url = None

        try:
            # 1. 이미지 업로드
            uploaded_image_url = await self._upload_image(image_data)

            if not uploaded_image_url:
                logger.warning("Image upload failed")
                return EnhancedSearchResponse(
                    success=False,
                    results=[],
                    search_links=self._generate_all_search_links(None),
                    error="이미지 업로드 실패"
                )

            logger.info(f"Image uploaded: {uploaded_image_url}")

            # 2. 병렬로 검색 + 원본 얼굴 임베딩 동시 추출
            import numpy as np

            embedding_task = asyncio.create_task(
                asyncio.to_thread(self._extract_embedding_sync, image_data)
            ) if self.face_recognition else None

            search_tasks = [
                self._search_serpapi_lens(uploaded_image_url),
                self._search_serpapi_google_images(uploaded_image_url),
                self._search_yandex(uploaded_image_url),
            ]

            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            for i, res in enumerate(search_results):
                if isinstance(res, Exception):
                    logger.warning(f"Search {i} failed: {res}")
                elif res:
                    results.extend(res)

            # 3. 중복 제거
            results = self._deduplicate_results(results)

            # 4. 플랫폼 분류
            results = self._classify_platforms(results)

            # 5. 얼굴 비교 (원본 임베딩은 이미 추출 완료)
            if embedding_task and results:
                original_embedding = await embedding_task
                if original_embedding:
                    results = await self._score_with_embedding(
                        np.array(original_embedding), results
                    )

            # 6. 점수순 정렬
            results.sort(key=lambda x: x.match_score, reverse=True)

            # 7. 얼굴 매칭 결과만 필터링 (match_score > 0) 또는 소셜미디어 결과
            social_platforms = {'instagram', 'facebook', 'linkedin', 'twitter', 'tiktok', 'vk'}
            filtered_results = [
                r for r in results
                if r.match_score > 30  # 얼굴 유사도 30% 이상
                or r.platform in social_platforms  # 소셜미디어 결과는 유지
            ]

            # 필터링 후 결과가 없으면 원본 중 소셜미디어만
            if not filtered_results:
                filtered_results = [r for r in results if r.platform in social_platforms]

            # 8. 수동 검색 링크 생성
            search_links = self._generate_all_search_links(uploaded_image_url)

            logger.info(f"Enhanced search complete: {len(filtered_results)} results (filtered from {len(results)})")

            return EnhancedSearchResponse(
                success=True,
                results=filtered_results[:30],  # 최대 30개
                search_links=search_links,
                uploaded_image_url=uploaded_image_url
            )

        except Exception as e:
            logger.error(f"Enhanced search failed: {e}", exc_info=True)
            return EnhancedSearchResponse(
                success=False,
                results=[],
                search_links=self._generate_all_search_links(uploaded_image_url),
                error=str(e)
            )

    async def _upload_image(self, image_data: bytes) -> str | None:
        """이미지 업로드 (여러 서비스 동시 시도, 먼저 성공한 것 사용)"""
        try:
            tasks = [
                asyncio.create_task(self._safe_upload(self._upload_to_imgbb, image_data)),
                asyncio.create_task(self._safe_upload(self._upload_to_catbox, image_data)),
                asyncio.create_task(self._safe_upload(self._upload_to_litterbox, image_data)),
            ]

            # 먼저 성공한 결과 반환, 나머지 취소
            while tasks:
                done, tasks_set = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(tasks_set)
                for task in done:
                    result = task.result()
                    if result:
                        for t in tasks:
                            t.cancel()
                        return result

            return None
        except Exception:
            return None

    async def _safe_upload(self, upload_fn, image_data: bytes) -> str | None:
        """업로드 함수를 안전하게 실행"""
        try:
            return await upload_fn(image_data)
        except Exception as e:
            logger.debug(f"Upload failed: {e}")
            return None

    async def _upload_to_imgbb(self, image_data: bytes) -> str | None:
        """imgbb 업로드"""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "https://api.imgbb.com/1/upload",
                    data={
                        "key": "b4e63b7398cf5d7db0248937c931a004",
                        "image": base64.b64encode(image_data).decode('utf-8')
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        return data["data"]["url"]
        except Exception as e:
            logger.debug(f"imgbb upload failed: {e}")
        return None

    async def _upload_to_catbox(self, image_data: bytes) -> str | None:
        """catbox 업로드"""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "https://catbox.moe/user/api.php",
                    data={"reqtype": "fileupload"},
                    files={"fileToUpload": ("image.jpg", image_data, "image/jpeg")}
                )
                if response.status_code == 200:
                    url = response.text.strip()
                    if url.startswith('http'):
                        return url
        except Exception as e:
            logger.debug(f"catbox upload failed: {e}")
        return None

    async def _upload_to_litterbox(self, image_data: bytes) -> str | None:
        """litterbox 업로드 (24시간 임시)"""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "https://litterbox.catbox.moe/resources/internals/api.php",
                    data={"reqtype": "fileupload", "time": "24h"},
                    files={"fileToUpload": ("image.jpg", image_data, "image/jpeg")}
                )
                if response.status_code == 200:
                    url = response.text.strip()
                    if url.startswith('http'):
                        return url
        except Exception as e:
            logger.debug(f"litterbox upload failed: {e}")
        return None

    async def _search_serpapi_lens(self, image_url: str) -> list[EnhancedSearchResult]:
        """SerpApi Google Lens 검색 (URL 사용)"""
        results = []

        if not self.settings.serpapi_key or not image_url:
            logger.debug("SerpApi key not configured or no image URL")
            return results

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    "https://serpapi.com/search",
                    params={
                        "engine": "google_lens",
                        "url": image_url,
                        "api_key": self.settings.serpapi_key
                    }
                )

                if response.status_code == 200:
                    data = response.json()

                    # Visual matches
                    for match in data.get("visual_matches", [])[:15]:
                        results.append(EnhancedSearchResult(
                            title=match.get("title", "유사 이미지"),
                            source_url=match.get("link", ""),
                            image_url=match.get("thumbnail", ""),
                            thumbnail_url=match.get("thumbnail"),
                            platform="other",
                            match_score=0,
                            source_engine="google_lens"
                        ))

                    # Knowledge graph (인물 정보)
                    for kg in data.get("knowledge_graph", [])[:5]:
                        if kg.get("link"):
                            results.append(EnhancedSearchResult(
                                title=kg.get("title", "관련 정보"),
                                source_url=kg.get("link", ""),
                                image_url=kg.get("thumbnail", ""),
                                thumbnail_url=kg.get("thumbnail"),
                                platform="other",
                                match_score=0,
                                source_engine="google_lens_kg"
                            ))

                    # Image sources (원본 출처)
                    for source in data.get("image_sources", [])[:10]:
                        results.append(EnhancedSearchResult(
                            title=source.get("title", "이미지 출처"),
                            source_url=source.get("link", ""),
                            image_url=source.get("image", ""),
                            thumbnail_url=source.get("thumbnail"),
                            platform="other",
                            match_score=0,
                            source_engine="google_lens_source"
                        ))

                    logger.info(f"SerpApi Google Lens found {len(results)} results")

        except Exception as e:
            logger.warning(f"SerpApi search failed: {e}")

        return results

    async def _search_yandex(self, image_url: str) -> list[EnhancedSearchResult]:
        """SerpApi Yandex 이미지 검색 (얼굴 인식 강력)"""
        results = []

        if not self.settings.serpapi_key or not image_url:
            return results

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    "https://serpapi.com/search",
                    params={
                        "engine": "yandex_images",
                        "url": image_url,
                        "api_key": self.settings.serpapi_key
                    }
                )

                if response.status_code == 200:
                    data = response.json()

                    # Yandex 이미지 결과
                    for img in data.get("image_results", [])[:15]:
                        results.append(EnhancedSearchResult(
                            title=img.get("title", "Yandex 결과"),
                            source_url=img.get("source", img.get("link", "")),
                            image_url=img.get("original", img.get("thumbnail", "")),
                            thumbnail_url=img.get("thumbnail"),
                            platform="other",
                            match_score=0,
                            source_engine="yandex"
                        ))

                    logger.info(f"Yandex found {len(results)} results")

        except Exception as e:
            logger.warning(f"Yandex search failed: {e}")

        return results

    async def _search_serpapi_google_images(self, image_url: str) -> list[EnhancedSearchResult]:
        """SerpApi Google 이미지 역검색"""
        results = []

        if not self.settings.serpapi_key:
            return results

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    "https://serpapi.com/search",
                    params={
                        "engine": "google_reverse_image",
                        "image_url": image_url,
                        "api_key": self.settings.serpapi_key
                    }
                )

                if response.status_code == 200:
                    data = response.json()

                    # Inline images
                    for img in data.get("inline_images", [])[:10]:
                        results.append(EnhancedSearchResult(
                            title=img.get("title", "유사 이미지"),
                            source_url=img.get("link", img.get("source", "")),
                            image_url=img.get("original", img.get("thumbnail", "")),
                            thumbnail_url=img.get("thumbnail"),
                            platform="other",
                            match_score=0,
                            source_engine="google_reverse"
                        ))

                    # Image results
                    for img in data.get("image_results", [])[:10]:
                        results.append(EnhancedSearchResult(
                            title=img.get("title", "관련 이미지"),
                            source_url=img.get("link", ""),
                            image_url=img.get("original", ""),
                            thumbnail_url=img.get("thumbnail"),
                            platform="other",
                            match_score=0,
                            source_engine="google_reverse"
                        ))

                    logger.info(f"SerpApi Google Reverse found {len(results)} results")

        except Exception as e:
            logger.warning(f"SerpApi Google Reverse failed: {e}")

        return results

    def _generate_all_search_links(self, image_url: str | None) -> list[dict]:
        """모든 검색 서비스 링크 생성"""
        links = []
        encoded_url = quote(image_url, safe='') if image_url else ""

        # 역이미지 검색 서비스
        reverse_image_services = [
            {
                "name": "Google Lens",
                "url": f"https://lens.google.com/uploadbyurl?url={encoded_url}" if image_url else "https://lens.google.com/",
                "icon": "https://www.google.com/favicon.ico",
                "category": "reverse_image",
                "description": "구글 렌즈로 이미지 검색",
                "priority": 1
            },
            {
                "name": "Yandex Images",
                "url": f"https://yandex.com/images/search?rpt=imageview&url={encoded_url}" if image_url else "https://yandex.com/images/",
                "icon": "https://yandex.com/favicon.ico",
                "category": "reverse_image",
                "description": "러시아 최대 검색엔진 - 얼굴 인식 강력",
                "priority": 2
            },
            {
                "name": "Bing Visual Search",
                "url": f"https://www.bing.com/images/search?view=detailv2&iss=sbi&q=imgurl:{encoded_url}" if image_url else "https://www.bing.com/visualsearch",
                "icon": "https://www.bing.com/favicon.ico",
                "category": "reverse_image",
                "description": "마이크로소프트 이미지 검색",
                "priority": 3
            },
            {
                "name": "TinEye",
                "url": f"https://tineye.com/search?url={encoded_url}" if image_url else "https://tineye.com/",
                "icon": "https://tineye.com/favicon.ico",
                "category": "reverse_image",
                "description": "이미지 원본 추적 전문",
                "priority": 4
            },
        ]

        # 얼굴 검색 전문 서비스
        face_search_services = [
            {
                "name": "PimEyes",
                "url": "https://pimeyes.com/",
                "icon": "https://pimeyes.com/favicon.ico",
                "category": "face_search",
                "description": "얼굴 인식 전문 검색 (유료)",
                "priority": 1
            },
            {
                "name": "FaceCheck.ID",
                "url": "https://facecheck.id/",
                "icon": "https://facecheck.id/favicon.ico",
                "category": "face_search",
                "description": "얼굴로 SNS 프로필 찾기",
                "priority": 2
            },
            {
                "name": "Search4faces",
                "url": "https://search4faces.com/",
                "icon": "https://search4faces.com/favicon.ico",
                "category": "face_search",
                "description": "VK/OK 소셜 미디어 얼굴 검색",
                "priority": 3
            },
            {
                "name": "Social Catfish",
                "url": "https://socialcatfish.com/",
                "icon": "https://socialcatfish.com/favicon.ico",
                "category": "face_search",
                "description": "로맨스 스캠 전문 조사",
                "priority": 4
            },
        ]

        # 스캠 신고/조회 서비스
        scam_check_services = [
            {
                "name": "Romance Scam (ScamDigger)",
                "url": "https://www.scamdigger.com/",
                "icon": "https://www.scamdigger.com/favicon.ico",
                "category": "scam_check",
                "description": "로맨스 스캐머 데이터베이스",
                "priority": 1
            },
            {
                "name": "ScamWarners",
                "url": "https://www.scamwarners.com/",
                "icon": "https://www.scamwarners.com/favicon.ico",
                "category": "scam_check",
                "description": "스캐머 경고 커뮤니티",
                "priority": 2
            },
            {
                "name": "Romance Scams Now",
                "url": "https://romancescamsnow.com/",
                "icon": "https://romancescamsnow.com/favicon.ico",
                "category": "scam_check",
                "description": "로맨스 스캠 정보 및 신고",
                "priority": 3
            },
            {
                "name": "Scamalytics",
                "url": "https://scamalytics.com/",
                "icon": "https://scamalytics.com/favicon.ico",
                "category": "scam_check",
                "description": "IP/이메일 사기 탐지",
                "priority": 4
            },
        ]

        links.extend(reverse_image_services)
        links.extend(face_search_services)
        links.extend(scam_check_services)

        return links

    def _deduplicate_results(self, results: list[EnhancedSearchResult]) -> list[EnhancedSearchResult]:
        """중복 제거"""
        seen_urls = set()
        unique = []

        for result in results:
            # URL에서 도메인+경로로 키 생성 (image_url이 dict일 수 있으므로 타입 체크)
            source = result.source_url if isinstance(result.source_url, str) else None
            image = result.image_url if isinstance(result.image_url, str) else None
            key = source.split('?')[0] if source else image
            if key and key not in seen_urls:
                seen_urls.add(key)
                unique.append(result)

        return unique

    def _classify_platforms(self, results: list[EnhancedSearchResult]) -> list[EnhancedSearchResult]:
        """플랫폼 분류"""
        platform_patterns = {
            'instagram': ['instagram.com', 'cdninstagram.com'],
            'facebook': ['facebook.com', 'fb.com', 'fbcdn.net'],
            'linkedin': ['linkedin.com', 'licdn.com'],
            'twitter': ['twitter.com', 'x.com', 'twimg.com'],
            'tiktok': ['tiktok.com', 'tiktokcdn.com'],
            'vk': ['vk.com', 'vkontakte.ru', 'userapi.com'],
            'youtube': ['youtube.com', 'youtu.be', 'ytimg.com'],
            'pinterest': ['pinterest.com', 'pinimg.com'],
            'reddit': ['reddit.com', 'redd.it'],
            'tumblr': ['tumblr.com'],
        }

        for result in results:
            # image_url이 dict일 수 있으므로 타입 체크
            source_url = result.source_url if isinstance(result.source_url, str) else ""
            image_url = result.image_url if isinstance(result.image_url, str) else ""
            url_lower = (source_url + image_url).lower()
            for platform, patterns in platform_patterns.items():
                if any(pattern in url_lower for pattern in patterns):
                    result.platform = platform
                    break

        return results

    async def _score_with_embedding(
        self,
        original_arr,
        results: list[EnhancedSearchResult]
    ) -> list[EnhancedSearchResult]:
        """미리 추출된 원본 임베딩으로 매치 스코어 계산"""
        import numpy as np

        try:
            semaphore = asyncio.Semaphore(5)

            # 유효한 URL이 있는 결과만, 최대 10개만 스코어링
            scoreable = [
                r for r in results
                if (isinstance(r.thumbnail_url, str) and r.thumbnail_url.startswith('http'))
                or (isinstance(r.image_url, str) and r.image_url.startswith('http'))
            ][:10]

            async def score_one(result: EnhancedSearchResult) -> None:
                async with semaphore:
                    try:
                        img_url = (
                            result.thumbnail_url if isinstance(result.thumbnail_url, str) and result.thumbnail_url.startswith('http')
                            else result.image_url
                        )

                        async with httpx.AsyncClient(timeout=5.0) as client:
                            response = await client.get(img_url, headers=self.headers)
                        if response.status_code != 200:
                            return

                        result_embedding = await asyncio.to_thread(
                            self._extract_embedding_sync, response.content
                        )
                        if result_embedding:
                            arr2 = np.array(result_embedding)
                            distance = float(np.linalg.norm(original_arr - arr2))
                            result.match_score = max(0, min(100, (1 - distance / 2) * 100))

                    except Exception as e:
                        logger.debug(f"Score calculation failed: {e}")

            # 전체 스코어링에 10초 제한
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(score_one(r) for r in scoreable)),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("Face match scoring timed out after 10s, returning partial results")

        except Exception as e:
            logger.warning(f"Match score error: {e}")

        return results

    def _extract_embedding_sync(self, image_data: bytes) -> list[float] | None:
        """동기 얼굴 임베딩 추출 (to_thread용)"""
        try:
            import io as _io
            from PIL import Image as _Image
            image = _Image.open(_io.BytesIO(image_data))
            import numpy as _np
            image_array = _np.array(image)
            from deepface import DeepFace
            embeddings = DeepFace.represent(
                img_path=image_array,
                model_name=self.face_recognition.model_name,
                enforce_detection=False
            )
            if embeddings and len(embeddings) > 0:
                return embeddings[0]["embedding"]
            return None
        except Exception as e:
            logger.debug(f"Embedding extraction failed: {e}")
            return None


class EnhancedSocialMediaSearcher:
    """강화된 소셜 미디어 검색"""

    def generate_search_links(self, name: str) -> list[dict]:
        """이름으로 소셜 미디어 검색 링크 생성"""
        encoded_name = quote(name)

        return [
            # 주요 SNS
            {
                "platform": "instagram",
                "name": "Instagram",
                "url": f"https://www.instagram.com/explore/search/keyword/?q={encoded_name}",
                "icon": "https://ui-avatars.com/api/?name=IG&background=E4405F&color=fff",
                "description": f"'{name}' Instagram 검색"
            },
            {
                "platform": "facebook",
                "name": "Facebook",
                "url": f"https://www.facebook.com/search/people/?q={encoded_name}",
                "icon": "https://ui-avatars.com/api/?name=FB&background=1877F2&color=fff",
                "description": f"'{name}' Facebook 검색"
            },
            {
                "platform": "linkedin",
                "name": "LinkedIn",
                "url": f"https://www.linkedin.com/search/results/people/?keywords={encoded_name}",
                "icon": "https://ui-avatars.com/api/?name=IN&background=0A66C2&color=fff",
                "description": f"'{name}' LinkedIn 검색"
            },
            {
                "platform": "twitter",
                "name": "X (Twitter)",
                "url": f"https://x.com/search?q={encoded_name}&f=user",
                "icon": "https://ui-avatars.com/api/?name=X&background=000000&color=fff",
                "description": f"'{name}' X 검색"
            },
            {
                "platform": "tiktok",
                "name": "TikTok",
                "url": f"https://www.tiktok.com/search?q={encoded_name}",
                "icon": "https://ui-avatars.com/api/?name=TT&background=000000&color=fff",
                "description": f"'{name}' TikTok 검색"
            },
            {
                "platform": "youtube",
                "name": "YouTube",
                "url": f"https://www.youtube.com/results?search_query={encoded_name}",
                "icon": "https://ui-avatars.com/api/?name=YT&background=FF0000&color=fff",
                "description": f"'{name}' YouTube 검색"
            },
            # 데이팅 앱/사이트
            {
                "platform": "dating",
                "name": "Dating Sites Search",
                "url": f"https://www.google.com/search?q={encoded_name}+site:tinder.com+OR+site:bumble.com+OR+site:hinge.co+OR+site:match.com",
                "icon": "https://ui-avatars.com/api/?name=DT&background=FE3C72&color=fff",
                "description": "데이팅 사이트에서 검색"
            },
            # 러시아/동유럽 SNS (스캠 흔함)
            {
                "platform": "vk",
                "name": "VK (VKontakte)",
                "url": f"https://vk.com/search?c[section]=people&c[q]={encoded_name}",
                "icon": "https://ui-avatars.com/api/?name=VK&background=4C75A3&color=fff",
                "description": "러시아 최대 SNS"
            },
            {
                "platform": "ok",
                "name": "Odnoklassniki",
                "url": f"https://ok.ru/search?query={encoded_name}",
                "icon": "https://ui-avatars.com/api/?name=OK&background=EE8208&color=fff",
                "description": "러시아 SNS"
            },
            # 아프리카 (나이지리아 스캠 많음)
            {
                "platform": "google_ng",
                "name": "Google Nigeria",
                "url": f"https://www.google.com.ng/search?q={encoded_name}",
                "icon": "https://ui-avatars.com/api/?name=NG&background=008751&color=fff",
                "description": "나이지리아 구글 검색"
            },
            # 구글 일반 검색
            {
                "platform": "google",
                "name": "Google",
                "url": f"https://www.google.com/search?q=\"{encoded_name}\"",
                "icon": "https://ui-avatars.com/api/?name=G&background=4285F4&color=fff",
                "description": f"'{name}' 구글 검색"
            },
            # 전화번호/이메일 조회
            {
                "platform": "truecaller",
                "name": "Truecaller",
                "url": f"https://www.truecaller.com/search/kr/{encoded_name}",
                "icon": "https://ui-avatars.com/api/?name=TC&background=0FA9E6&color=fff",
                "description": "전화번호 조회"
            },
        ]

    def generate_phone_search_links(self, phone: str) -> list[dict]:
        """전화번호로 검색 링크 생성"""
        clean_phone = ''.join(filter(str.isdigit, phone))

        return [
            {
                "platform": "truecaller",
                "name": "Truecaller",
                "url": f"https://www.truecaller.com/search/{clean_phone}",
                "icon": "https://ui-avatars.com/api/?name=TC&background=0FA9E6&color=fff",
                "description": "전화번호 소유자 확인"
            },
            {
                "platform": "google_phone",
                "name": "Google Phone Search",
                "url": f"https://www.google.com/search?q=\"{phone}\"",
                "icon": "https://ui-avatars.com/api/?name=G&background=4285F4&color=fff",
                "description": "전화번호 구글 검색"
            },
            {
                "platform": "whoscall",
                "name": "Whoscall",
                "url": f"https://whoscall.com/search/{clean_phone}",
                "icon": "https://ui-avatars.com/api/?name=WC&background=00C853&color=fff",
                "description": "스팸 전화 확인"
            },
        ]

    def generate_email_search_links(self, email: str) -> list[dict]:
        """이메일로 검색 링크 생성"""
        encoded_email = quote(email)

        return [
            {
                "platform": "haveibeenpwned",
                "name": "Have I Been Pwned",
                "url": f"https://haveibeenpwned.com/account/{encoded_email}",
                "icon": "https://ui-avatars.com/api/?name=HI&background=2A6496&color=fff",
                "description": "이메일 유출 확인"
            },
            {
                "platform": "google_email",
                "name": "Google Email Search",
                "url": f"https://www.google.com/search?q=\"{encoded_email}\"",
                "icon": "https://ui-avatars.com/api/?name=G&background=4285F4&color=fff",
                "description": "이메일 구글 검색"
            },
            {
                "platform": "epieos",
                "name": "Epieos",
                "url": f"https://epieos.com/?q={encoded_email}",
                "icon": "https://ui-avatars.com/api/?name=EP&background=6C5CE7&color=fff",
                "description": "이메일 정보 조회"
            },
        ]
