"""
의존성 주입 (Dependency Injection)
FastAPI의 Depends를 통해 유스케이스와 서비스를 주입
"""
from functools import lru_cache

from src.application.chat import AnalyzeChatUseCase, ChatbotUseCase, GetPatternsUseCase
from src.application.deepfake import AnalyzeImageUseCase, AnalyzeVideoUseCase
from src.application.profile import ProfileSearchUseCase, ReportScammerUseCase
from src.infrastructure.ai import FaceRecognitionService
from src.infrastructure.external import (
    ImageSearchScraper,
    OpenAIService,
    SightengineService,
    SocialMediaSearcher,
)
from src.infrastructure.persistence import JsonScammerRepository, QdrantScamRepository, ScammerNetworkRepository

# ==================== 서비스 싱글톤 ====================

@lru_cache
def get_face_recognition_service() -> FaceRecognitionService:
    return FaceRecognitionService()


@lru_cache
def get_sightengine_service() -> SightengineService:
    return SightengineService()


@lru_cache
def get_openai_service() -> OpenAIService:
    return OpenAIService()


@lru_cache
def get_scammer_repository() -> JsonScammerRepository:
    return JsonScammerRepository()


@lru_cache
def get_qdrant_repository() -> QdrantScamRepository:
    return QdrantScamRepository()


@lru_cache
def get_scammer_network_repository() -> ScammerNetworkRepository:
    return ScammerNetworkRepository()


@lru_cache
def get_image_search_scraper() -> ImageSearchScraper:
    return ImageSearchScraper(face_recognition_service=get_face_recognition_service())


@lru_cache
def get_social_media_searcher() -> SocialMediaSearcher:
    return SocialMediaSearcher()


# ==================== Profile 유스케이스 ====================

def get_profile_search_use_case() -> ProfileSearchUseCase:
    return ProfileSearchUseCase(
        face_recognition=get_face_recognition_service(),
        scammer_repository=get_scammer_repository(),
        image_search_scraper=get_image_search_scraper(),
        social_media_searcher=get_social_media_searcher()
    )


def get_report_scammer_use_case() -> ReportScammerUseCase:
    return ReportScammerUseCase(
        face_recognition=get_face_recognition_service(),
        scammer_repository=get_scammer_repository()
    )


# ==================== Deepfake 유스케이스 ====================

def get_analyze_image_use_case() -> AnalyzeImageUseCase:
    return AnalyzeImageUseCase(
        sightengine=get_sightengine_service(),
        openai=get_openai_service()
    )


def get_analyze_video_use_case() -> AnalyzeVideoUseCase:
    return AnalyzeVideoUseCase(
        sightengine=get_sightengine_service()
    )


# ==================== Chat 유스케이스 ====================

def get_analyze_chat_use_case() -> AnalyzeChatUseCase:
    return AnalyzeChatUseCase(
        ai_service=get_openai_service(),
        qdrant_repo=get_qdrant_repository()
    )


def get_chatbot_use_case() -> ChatbotUseCase:
    return ChatbotUseCase(
        ai_service=get_openai_service()
    )


def get_patterns_use_case() -> GetPatternsUseCase:
    return GetPatternsUseCase()


# ==================== 초기화 함수 ====================

async def initialize_services():
    """모든 서비스 초기화"""
    import logging
    logger = logging.getLogger(__name__)

    face_recognition = get_face_recognition_service()
    await face_recognition.initialize()

    # OpenAI 초기화 (실패해도 서버는 계속 동작 - 폴백 사용)
    try:
        openai_service = get_openai_service()
        await openai_service.initialize()
    except Exception as e:
        logger.warning(f"OpenAI 초기화 실패 (폴백 사용): {e}")

    scammer_repo = get_scammer_repository()
    await scammer_repo.initialize()

    # QdrantScamRepository 초기화 (대화 패턴 벡터 검색)
    try:
        qdrant_repo = get_qdrant_repository()
        await qdrant_repo.initialize()
    except Exception as e:
        logger.warning(f"Qdrant 초기화 실패 (서버는 계속 동작): {e}")

    # ScammerNetworkRepository 초기화 (스캐머 네트워크 분석)
    try:
        network_repo = get_scammer_network_repository()
        await network_repo.initialize()
    except Exception as e:
        logger.warning(f"Neo4j 스캐머 네트워크 초기화 실패 (서버는 계속 동작): {e}")
