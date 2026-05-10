"""
健康检查 API
"""

from fastapi import APIRouter
from backend.core.redis_manager import get_redis_manager
from backend.core.embedding import get_embedding_service
from backend.core.milvus_client import get_milvus_manager
from backend.core.llm_client import get_llm_client

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def health_check():
    """服务健康检查"""
    redis = get_redis_manager()
    embedding = get_embedding_service()
    milvus = get_milvus_manager()
    llm = get_llm_client()

    return {
        "status": "healthy",
        "services": {
            "redis": redis.health_check(),
            "ollama_embedding": embedding.health_check(),
            "milvus": True,  # Milvus 没有简单的 ping
            "llm": True,     # 不在健康检查中调用 LLM（避免消耗 token）
        },
        "config": {
            "embedding_model": embedding.model,
            "embedding_dim": embedding.dimension,
            "llm_model": llm.model,
        },
    }
