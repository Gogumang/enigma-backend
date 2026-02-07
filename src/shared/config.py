from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 설정"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    port: int = 4000

    # Sightengine
    sightengine_api_user: str = ""
    sightengine_api_secret: str = ""

    # OpenAI
    openai_api_key: str = ""

    # Qdrant (대화 패턴 벡터 검색)
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # Neo4j (스캐머 네트워크 분석)
    neo4j_uri: str = ""
    neo4j_username: str = ""
    neo4j_password: str = ""

    # SerpApi (역이미지 검색)
    serpapi_key: str = ""

    # Google Safe Browsing
    google_safe_browsing_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
