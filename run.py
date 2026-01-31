#!/usr/bin/env python3
"""
Love Guard API 서버 실행 스크립트
"""
import uvicorn
from src.shared.config import get_settings

if __name__ == "__main__":
    settings = get_settings()

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
        log_level="info"
    )
