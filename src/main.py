import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.interfaces.api import (
    chat_router,
    deepfake_router,
    fraud_router,
    initialize_services,
    network_router,
    profile_router,
    training_router,
    url_router,
)
from src.shared.config import get_settings
from src.shared.exceptions import DomainException

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    # 시작 시
    logger.info("Starting Love Guard API...")
    await initialize_services()
    logger.info("All services initialized")

    yield

    # 종료 시
    logger.info("Shutting down Love Guard API...")


# FastAPI 앱 생성
app = FastAPI(
    title="Love Guard API",
    description="로맨스 스캠 예방 API - DDD Architecture",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# CORS 설정
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.cors_origin,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 예외 핸들러
@app.exception_handler(DomainException)
async def domain_exception_handler(_request: Request, exc: DomainException):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": exc.message,
            "code": exc.code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(_request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "code": "INTERNAL_ERROR"
        }
    )


# 라우터 등록 (prefix: /api)
app.include_router(profile_router, prefix="/api")
app.include_router(deepfake_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(fraud_router, prefix="/api")
app.include_router(network_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(url_router, prefix="/api")


# 헬스체크
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Love Guard API",
        "version": "2.0.0"
    }


# 루트
@app.get("/")
async def root():
    return {
        "message": "Welcome to Love Guard API",
        "docs": "/api/docs"
    }


if __name__ == "__main__":
    import uvicorn

    print(f"""
    ╔═══════════════════════════════════════════╗
    ║         Love Guard API v2.0.0             ║
    ║         DDD Architecture + FastAPI        ║
    ╠═══════════════════════════════════════════╣
    ║  Server: http://localhost:{settings.port}            ║
    ║  Docs:   http://localhost:{settings.port}/api/docs   ║
    ╚═══════════════════════════════════════════╝
    """)

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=True,
        log_level="info",
    )
