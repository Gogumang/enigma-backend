"""
역이미지 검색 스크래퍼
- Yandex, Google, Bing 등에서 검색 결과를 가져옴
- 얼굴 비교로 일치율 계산
"""
import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass
from urllib.parse import quote, urljoin

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class ImageSearchResult:
    """검색된 이미지 결과"""
    title: str
    source_url: str
    image_url: str
    thumbnail_url: str | None
    platform: str  # instagram, facebook, linkedin, twitter, other
    match_score: float  # 0-100


@dataclass
class ImageSearchResponse:
    """검색 응답"""
    success: bool
    results: list[ImageSearchResult]
    uploaded_image_url: str | None = None
    error: str | None = None


class ImageSearchScraper:
    """역이미지 검색 스크래퍼"""

    def __init__(self, face_recognition_service=None):
        self.face_recognition = face_recognition_service
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    async def search(self, image_data: bytes) -> ImageSearchResponse:
        """이미지로 역이미지 검색 수행"""
        try:
            # 1. 이미지 업로드
            image_url = await self._upload_image(image_data)
            if not image_url:
                logger.warning("Image upload failed, trying fallback")
                return ImageSearchResponse(
                    success=False,
                    results=[],
                    error="이미지 업로드 실패"
                )

            logger.info(f"Image uploaded: {image_url}")

            # 2. 여러 검색 엔진에서 병렬로 검색
            results = []

            # 동시에 여러 검색 엔진 시도
            search_tasks = [
                self._search_yandex(image_url),
                self._search_bing(image_url),
                self._search_tineye_style(image_url),
            ]

            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            for i, res in enumerate(search_results):
                if isinstance(res, Exception):
                    logger.warning(f"Search engine {i} failed: {res}")
                elif res:
                    results.extend(res)
                    logger.info(f"Search engine {i} found {len(res)} results")

            # 3. 중복 제거 및 정렬
            results = self._deduplicate_results(results)

            # 4. 플랫폼별 분류
            results = self._classify_by_platform(results)

            # 5. 원본 이미지와 비교하여 일치율 계산 (썸네일이 있는 경우만)
            if self.face_recognition and results:
                results = await self._calculate_match_scores(image_data, results)

            # 6. 결과가 없으면 수동 검색 링크 제공
            if not results:
                logger.info("No automated results found, providing manual search links")
                results = self._generate_manual_search_links(image_url)

            logger.info(f"Total found: {len(results)} results after deduplication")

            return ImageSearchResponse(
                success=True,
                results=results,
                uploaded_image_url=image_url
            )

        except Exception as e:
            logger.error(f"Image search failed: {e}", exc_info=True)
            return ImageSearchResponse(
                success=False,
                results=[],
                error=str(e)
            )

    async def _upload_image(self, image_data: bytes) -> str | None:
        """이미지를 무료 호스팅에 업로드 (여러 서비스 시도)"""

        # 1. imgbb 사용 (가장 안정적)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                response = await client.post(
                    "https://api.imgbb.com/1/upload",
                    data={
                        "key": "b4e63b7398cf5d7db0248937c931a004",
                        "image": image_base64
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        url = data["data"]["url"]
                        logger.info(f"imgbb upload success: {url}")
                        return url
                logger.warning(f"imgbb response: {response.status_code}")
        except Exception as e:
            logger.warning(f"imgbb upload failed: {e}")

        # 2. catbox.moe (신뢰성 높음)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"fileToUpload": ("image.jpg", image_data, "image/jpeg")}
                response = await client.post(
                    "https://catbox.moe/user/api.php",
                    data={"reqtype": "fileupload"},
                    files=files
                )
                if response.status_code == 200:
                    url = response.text.strip()
                    if url.startswith('http'):
                        logger.info(f"catbox upload success: {url}")
                        return url
        except Exception as e:
            logger.warning(f"catbox upload failed: {e}")

        # 3. litterbox (catbox의 임시 파일 서비스)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"fileToUpload": ("image.jpg", image_data, "image/jpeg")}
                response = await client.post(
                    "https://litterbox.catbox.moe/resources/internals/api.php",
                    data={"reqtype": "fileupload", "time": "24h"},
                    files=files
                )
                if response.status_code == 200:
                    url = response.text.strip()
                    if url.startswith('http'):
                        logger.info(f"litterbox upload success: {url}")
                        return url
        except Exception as e:
            logger.warning(f"litterbox upload failed: {e}")

        # 4. 0x0.st 백업
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"file": ("image.jpg", image_data, "image/jpeg")}
                response = await client.post("https://0x0.st", files=files)
                if response.status_code == 200:
                    url = response.text.strip()
                    logger.info(f"0x0.st upload success: {url}")
                    return url
        except Exception as e:
            logger.warning(f"0x0.st upload failed: {e}")

        logger.error("All image upload services failed")
        return None

    async def _search_yandex(self, image_url: str) -> list[ImageSearchResult]:
        """Yandex 역이미지 검색 - 실제 이미지 결과 추출"""
        results = []
        try:
            encoded_url = quote(image_url, safe='')
            search_url = f"https://yandex.com/images/search?rpt=imageview&url={encoded_url}"

            # 더 현실적인 헤더
            yandex_headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"macOS"',
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
            }

            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(search_url, headers=yandex_headers)

                logger.info(f"Yandex response status: {response.status_code}")

                if response.status_code == 200:
                    html = response.text
                    soup = BeautifulSoup(html, 'html.parser')

                    # 방법 1: JSON 데이터에서 추출 (더 신뢰성 있음)
                    json_results = self._extract_yandex_json(html)
                    if json_results:
                        results.extend(json_results)
                        logger.info(f"Yandex JSON method found {len(json_results)} results")

                    # 방법 2: HTML에서 직접 추출
                    if len(results) < 5:
                        html_results = self._extract_yandex_html(soup)
                        results.extend(html_results)
                        logger.info(f"Yandex HTML method found {len(html_results)} results")

                    # 방법 3: 이미지 썸네일 직접 찾기
                    if len(results) < 5:
                        img_results = self._extract_images_from_html(soup, search_url)
                        results.extend(img_results)
                        logger.info(f"Yandex image method found {len(img_results)} results")

            logger.info(f"Yandex found {len(results)} total results")

        except Exception as e:
            logger.warning(f"Yandex search failed: {e}")

        return results

    async def _search_bing(self, image_url: str) -> list[ImageSearchResult]:
        """Bing 역이미지 검색"""
        results = []
        try:
            encoded_url = quote(image_url, safe='')
            search_url = f"https://www.bing.com/images/search?view=detailv2&iss=sbi&q=imgurl:{encoded_url}"

            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(search_url, headers=self.headers)

                if response.status_code == 200:
                    html = response.text
                    soup = BeautifulSoup(html, 'html.parser')

                    # Bing 이미지 결과 추출
                    for item in soup.select('.imgpt, .img_cont, .mimg')[:15]:
                        try:
                            # 이미지 찾기
                            img = item.find('img')
                            if not img:
                                continue

                            img_src = img.get('src') or img.get('data-src') or img.get('data-src2', '')
                            if not img_src or img_src.startswith('data:'):
                                continue

                            # URL 정규화
                            if img_src.startswith('//'):
                                img_src = f"https:{img_src}"
                            elif not img_src.startswith('http'):
                                img_src = urljoin('https://www.bing.com', img_src)

                            # 링크 찾기
                            parent_link = item.find_parent('a') or item.find('a')
                            href = parent_link.get('href', '') if parent_link else ''

                            # m= 파라미터에서 실제 URL 추출 시도
                            if 'm=' in href:
                                import urllib.parse
                                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                                if 'm' in parsed:
                                    href = parsed['m'][0]

                            if href and not href.startswith('http'):
                                href = urljoin('https://www.bing.com', href)

                            # 제목
                            title = img.get('alt', '') or '유사 이미지'

                            if img_src:
                                results.append(ImageSearchResult(
                                    title=title,
                                    source_url=href or search_url,
                                    image_url=img_src,
                                    thumbnail_url=img_src,
                                    platform="other",
                                    match_score=0
                                ))
                        except Exception as e:
                            logger.debug(f"Failed to parse Bing item: {e}")
                            continue

            logger.info(f"Bing found {len(results)} results")

        except Exception as e:
            logger.warning(f"Bing search failed: {e}")

        return results

    async def _search_tineye_style(self, image_url: str) -> list[ImageSearchResult]:
        """Google Lens 스타일 검색 (구글 이미지 검색)"""
        results = []
        try:
            encoded_url = quote(image_url, safe='')
            # Google 이미지 검색 URL
            search_url = f"https://www.google.com/searchbyimage?image_url={encoded_url}"

            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(search_url, headers={
                    **self.headers,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                })

                if response.status_code == 200:
                    html = response.text
                    soup = BeautifulSoup(html, 'html.parser')

                    # 유사 이미지 섹션 찾기
                    for item in soup.select('div[data-ri], .isv-r, .rg_bx')[:15]:
                        try:
                            img = item.find('img')
                            if not img:
                                continue

                            img_src = img.get('src') or img.get('data-src', '')
                            if not img_src or img_src.startswith('data:'):
                                continue

                            if img_src.startswith('//'):
                                img_src = f"https:{img_src}"

                            link = item.find('a')
                            href = link.get('href', '') if link else ''

                            title = img.get('alt', '') or '유사 이미지'

                            if img_src:
                                results.append(ImageSearchResult(
                                    title=title,
                                    source_url=href or search_url,
                                    image_url=img_src,
                                    thumbnail_url=img_src,
                                    platform="other",
                                    match_score=0
                                ))
                        except Exception as e:
                            continue

            logger.info(f"Google style search found {len(results)} results")

        except Exception as e:
            logger.warning(f"Google style search failed: {e}")

        return results

    def _extract_yandex_json(self, html: str) -> list[ImageSearchResult]:
        """Yandex 페이지의 JSON 데이터에서 결과 추출"""
        results = []
        try:
            # Yandex는 페이지에 JSON 데이터를 포함시킴
            patterns = [
                r'"serpList":\s*(\[.*?\])',
                r'"items":\s*(\[.*?\])',
                r'var defined\s*=\s*({.*?});',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, html, re.DOTALL)
                for match in matches:
                    try:
                        data = json.loads(match)
                        if isinstance(data, list):
                            for item in data[:10]:
                                if isinstance(item, dict):
                                    thumb = item.get('thumb', {})
                                    thumb_url = thumb.get('url') if isinstance(thumb, dict) else None

                                    result = ImageSearchResult(
                                        title=item.get('title', '') or item.get('snippet', '') or '유사 이미지',
                                        source_url=item.get('url', '') or item.get('pageUrl', ''),
                                        image_url=item.get('origUrl', '') or item.get('imageUrl', ''),
                                        thumbnail_url=thumb_url or item.get('thumbUrl', ''),
                                        platform="other",
                                        match_score=0
                                    )
                                    if result.source_url:
                                        results.append(result)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.debug(f"JSON extraction failed: {e}")

        return results

    def _extract_yandex_html(self, soup: BeautifulSoup) -> list[ImageSearchResult]:
        """HTML에서 직접 결과 추출"""
        results = []

        # 다양한 선택자 시도
        selectors = [
            'div.serp-item',
            'div.other-sites__item',
            'a.other-sites__preview-link',
            'div.CbirSites-Item',
            'div.cbir-similar__item',
        ]

        for selector in selectors:
            items = soup.select(selector)
            for item in items[:10]:
                try:
                    # 링크 찾기
                    link = item.find('a')
                    href = link.get('href', '') if link else ''

                    # 이미지 찾기
                    img = item.find('img')
                    img_src = ''
                    if img:
                        img_src = img.get('src', '') or img.get('data-src', '') or img.get('data-original', '')

                    # 제목 찾기
                    title_elem = item.find(['span', 'div'], class_=re.compile(r'title|name|text'))
                    title = title_elem.get_text(strip=True) if title_elem else ''

                    if href or img_src:
                        # URL 정규화
                        if href and not href.startswith('http'):
                            href = f"https://yandex.com{href}"
                        if img_src and not img_src.startswith('http'):
                            img_src = f"https:{img_src}" if img_src.startswith('//') else f"https://yandex.com{img_src}"

                        results.append(ImageSearchResult(
                            title=title or "유사 이미지",
                            source_url=href,
                            image_url=img_src,
                            thumbnail_url=img_src,
                            platform="other",
                            match_score=0
                        ))
                except Exception as e:
                    logger.debug(f"Failed to parse item: {e}")
                    continue

        return results

    def _extract_images_from_html(self, soup: BeautifulSoup, base_url: str) -> list[ImageSearchResult]:
        """모든 이미지 태그에서 결과 추출"""
        results = []

        # 모든 이미지 찾기
        images = soup.find_all('img')
        seen_urls = set()

        for img in images:
            try:
                src = img.get('src', '') or img.get('data-src', '')

                # 작은 아이콘이나 로고 제외
                if not src or 'logo' in src.lower() or 'icon' in src.lower():
                    continue
                if 'avatar' in src.lower() and 'yandex' in src.lower():
                    continue

                # URL 정규화
                if not src.startswith('http'):
                    if src.startswith('//'):
                        src = f"https:{src}"
                    else:
                        continue

                # 중복 제거
                if src in seen_urls:
                    continue
                seen_urls.add(src)

                # 부모 링크 찾기
                parent_link = img.find_parent('a')
                href = parent_link.get('href', '') if parent_link else ''
                if href and not href.startswith('http'):
                    href = f"https://yandex.com{href}"

                # 제목 찾기
                alt = img.get('alt', '') or img.get('title', '')

                results.append(ImageSearchResult(
                    title=alt or "발견된 이미지",
                    source_url=href or base_url,
                    image_url=src,
                    thumbnail_url=src,
                    platform="other",
                    match_score=0
                ))

                if len(results) >= 15:
                    break

            except Exception as e:
                continue

        return results

    async def _calculate_match_scores(
        self,
        original_image: bytes,
        results: list[ImageSearchResult]
    ) -> list[ImageSearchResult]:
        """각 결과 이미지와 원본 비교하여 일치율 계산"""
        if not self.face_recognition:
            return results

        try:
            # 원본 이미지 임베딩 추출
            original_embedding = await self.face_recognition.extract_embedding(original_image)
            if not original_embedding:
                return results

            import numpy as np

            async with httpx.AsyncClient(timeout=10.0) as client:
                for result in results:
                    try:
                        img_url = result.thumbnail_url or result.image_url
                        if not img_url or not img_url.startswith('http'):
                            continue

                        # 이미지 다운로드
                        response = await client.get(img_url, headers=self.headers)
                        if response.status_code != 200:
                            continue

                        # 얼굴 비교
                        result_embedding = await self.face_recognition.extract_embedding(response.content)
                        if result_embedding:
                            arr1 = np.array(original_embedding)
                            arr2 = np.array(result_embedding)
                            distance = float(np.linalg.norm(arr1 - arr2))
                            result.match_score = max(0, min(100, (1 - distance / 2) * 100))

                    except Exception as e:
                        logger.debug(f"Match score calculation failed: {e}")
                        continue

        except Exception as e:
            logger.warning(f"Match score calculation error: {e}")

        return results

    def _deduplicate_results(self, results: list[ImageSearchResult]) -> list[ImageSearchResult]:
        """중복 제거"""
        seen = set()
        unique = []

        for result in results:
            # URL에서 도메인+경로로 키 생성
            key = result.source_url.split('?')[0] if result.source_url else result.thumbnail_url
            if key and key not in seen:
                seen.add(key)
                unique.append(result)

        return unique

    def _classify_by_platform(self, results: list[ImageSearchResult]) -> list[ImageSearchResult]:
        """URL을 기반으로 플랫폼 분류"""
        platform_patterns = {
            'instagram': ['instagram.com', 'instagr.am', 'cdninstagram.com'],
            'facebook': ['facebook.com', 'fb.com', 'fb.me', 'fbcdn.net'],
            'linkedin': ['linkedin.com', 'licdn.com'],
            'twitter': ['twitter.com', 'x.com', 't.co', 'twimg.com'],
            'tiktok': ['tiktok.com', 'tiktokcdn.com'],
            'vk': ['vk.com', 'vkontakte.ru', 'userapi.com'],
        }

        for result in results:
            url_lower = (result.source_url + result.image_url).lower()
            for platform, patterns in platform_patterns.items():
                if any(pattern in url_lower for pattern in patterns):
                    result.platform = platform
                    break

        return results

    def _generate_manual_search_links(self, image_url: str) -> list[ImageSearchResult]:
        """수동 검색 링크 생성 (자동 검색 실패 시)"""
        encoded_url = quote(image_url, safe='')

        return [
            ImageSearchResult(
                title="Yandex에서 역이미지 검색",
                source_url=f"https://yandex.com/images/search?rpt=imageview&url={encoded_url}",
                image_url="https://yastatic.net/s3/home-static/_/37/37a02b5dc7a51abac55d8a5b6c865f0e.png",
                thumbnail_url="https://yastatic.net/s3/home-static/_/37/37a02b5dc7a51abac55d8a5b6c865f0e.png",
                platform="other",
                match_score=0
            ),
            ImageSearchResult(
                title="Google에서 역이미지 검색",
                source_url=f"https://lens.google.com/uploadbyurl?url={encoded_url}",
                image_url="https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_92x30dp.png",
                thumbnail_url="https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_92x30dp.png",
                platform="other",
                match_score=0
            ),
            ImageSearchResult(
                title="Bing에서 역이미지 검색",
                source_url=f"https://www.bing.com/images/search?view=detailv2&iss=sbi&q=imgurl:{encoded_url}",
                image_url="https://www.bing.com/sa/simg/favicon-trans-bg-blue-mg.ico",
                thumbnail_url="https://www.bing.com/sa/simg/favicon-trans-bg-blue-mg.ico",
                platform="other",
                match_score=0
            ),
            ImageSearchResult(
                title="TinEye에서 역이미지 검색",
                source_url=f"https://tineye.com/search?url={encoded_url}",
                image_url="https://tineye.com/images/widgets/mona.jpg",
                thumbnail_url="https://tineye.com/images/widgets/mona.jpg",
                platform="other",
                match_score=0
            ),
        ]


class SocialMediaSearcher:
    """소셜 미디어 프로필 검색"""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

    async def search_by_name(self, name: str) -> list[ImageSearchResult]:
        """이름으로 소셜 미디어 검색 링크 생성"""
        from urllib.parse import quote
        encoded_name = quote(name)

        return [
            ImageSearchResult(
                title=f"Instagram에서 '{name}' 검색",
                source_url=f"https://www.instagram.com/explore/search/keyword/?q={encoded_name}",
                image_url="",
                thumbnail_url=None,
                platform="instagram",
                match_score=0
            ),
            ImageSearchResult(
                title=f"Facebook에서 '{name}' 검색",
                source_url=f"https://www.facebook.com/search/people/?q={encoded_name}",
                image_url="",
                thumbnail_url=None,
                platform="facebook",
                match_score=0
            ),
            ImageSearchResult(
                title=f"LinkedIn에서 '{name}' 검색",
                source_url=f"https://www.linkedin.com/search/results/people/?keywords={encoded_name}",
                image_url="",
                thumbnail_url=None,
                platform="linkedin",
                match_score=0
            ),
            ImageSearchResult(
                title=f"X에서 '{name}' 검색",
                source_url=f"https://x.com/search?q={encoded_name}&f=user",
                image_url="",
                thumbnail_url=None,
                platform="twitter",
                match_score=0
            ),
        ]
