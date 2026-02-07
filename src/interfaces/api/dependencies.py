"""
의존성 주입 (Dependency Injection)
FastAPI의 Depends를 통해 유스케이스와 서비스를 주입
"""
import asyncio
from functools import lru_cache

from src.application.chat import AnalyzeChatUseCase
from src.application.deepfake import AnalyzeImageUseCase, AnalyzeVideoUseCase
from src.application.profile import ProfileSearchUseCase
from src.infrastructure.ai import FaceRecognitionService
from src.infrastructure.external import (
    ImageSearchScraper,
    OpenAIService,
    SightengineService,
    SocialMediaSearcher,
)
from src.infrastructure.persistence import (
    JsonScammerRepository,
    QdrantScamRepository,
    Neo4jRelationshipRepository,
    ScamReportRepository,
)

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
def get_relationship_repository() -> Neo4jRelationshipRepository:
    return Neo4jRelationshipRepository()


@lru_cache
def get_scam_report_repository() -> ScamReportRepository:
    return ScamReportRepository()


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


# ==================== Deepfake 유스케이스 ====================

def get_analyze_image_use_case() -> AnalyzeImageUseCase:
    return AnalyzeImageUseCase(
        sightengine=get_sightengine_service()
    )


def get_analyze_video_use_case() -> AnalyzeVideoUseCase:
    return AnalyzeVideoUseCase(
        sightengine=get_sightengine_service()
    )


# ==================== Chat 유스케이스 ====================

def get_analyze_chat_use_case() -> AnalyzeChatUseCase:
    return AnalyzeChatUseCase(
        ai_service=get_openai_service(),
        qdrant_repo=get_qdrant_repository(),
        relationship_repo=get_relationship_repository()
    )


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

    # Neo4jRelationshipRepository 초기화 (사용자 관계 분석)
    try:
        relationship_repo = get_relationship_repository()
        await relationship_repo.initialize()
    except Exception as e:
        logger.warning(f"Neo4j 관계 리포지토리 초기화 실패 (서버는 계속 동작): {e}")

    # ScamReportRepository 초기화 (사기 신고 저장)
    try:
        scam_report_repo = get_scam_report_repository()
        await scam_report_repo.initialize()
    except Exception as e:
        logger.warning(f"ScamReport 리포지토리 초기화 실패 (서버는 계속 동작): {e}")

    # AI 모델 프리로드 (첫 요청 cold start 방지, 실패해도 서버 동작)
    await _preload_ai_models(logger)


async def _preload_ai_models(logger):
    """UnivFD + EfficientViT 모델을 병렬로 미리 로드"""

    async def _load_univfd():
        try:
            from src.infrastructure.ai.univfd_detector import get_univfd_detector
            detector = get_univfd_detector()
            await asyncio.to_thread(detector._ensure_initialized)
            logger.info("UnivFD 모델 프리로드 완료")
        except Exception as e:
            logger.warning(f"UnivFD 프리로드 실패 (폴백 사용): {e}")

    async def _load_explainer():
        try:
            from src.infrastructure.ai.deepfake_explainer import get_deepfake_explainer_service
            explainer = get_deepfake_explainer_service()
            if explainer.is_available():
                await asyncio.to_thread(explainer._ensure_initialized)
                logger.info("EfficientViT 모델 프리로드 완료")
        except Exception as e:
            logger.warning(f"EfficientViT 프리로드 실패 (폴백 사용): {e}")

    await asyncio.gather(_load_univfd(), _load_explainer())
