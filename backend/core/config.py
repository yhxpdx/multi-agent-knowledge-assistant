from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # 火山引擎 Ark API
    ARK_API_KEY: str
    ARK_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/coding/v3"
    ARK_MODEL_ID: str = "doubao-seed-2-0-lite-260428"

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "document_chunks"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Ollama (local embedding)
    OLLAMA_HOST: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "bge-m3"
    EMBEDDING_DIM: int = 1024

    # Document processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_FILE_SIZE_MB: int = 20

    # Retrieval
    TOP_K: int = 5

    # Conversation
    MAX_HISTORY_TURNS: int = 10
    SESSION_TTL_SECONDS: int = 604800  # 7 days

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
