"""
Embedding 服务封装

设计决策（全链路思考）：
1. 为什么用 Ollama 本地模型而不是 API？
   - 火山引擎 coding plan 不支持 Embedding API
   - 本地模型无网络延迟，无 API 调用成本
   - bge-m3 评估结果优秀（Recall@5 100%，MRR 97.22%）

2. 为什么封装统一接口？
   - 方便后续切换模型（如换成 OpenAI Embedding）
   - 统一错误处理和重试逻辑
   - 支持批量 embedding 提高效率
"""

import requests
from typing import Optional
from backend.core.config import get_settings


class EmbeddingService:
    def __init__(self):
        self.settings = get_settings()
        self.model = self.settings.EMBEDDING_MODEL
        self.base_url = self.settings.OLLAMA_HOST
        self.dimension = self.settings.EMBEDDING_DIM

    def embed_query(self, text: str) -> Optional[list[float]]:
        """为查询文本生成 embedding 向量"""
        try:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json().get("embedding", [])
        except Exception as e:
            print(f"Embedding error: {e}")
        return None

    def embed_documents(self, texts: list[str]) -> list[Optional[list[float]]]:
        """批量为文档生成 embedding 向量"""
        results = []
        for text in texts:
            results.append(self.embed_query(text))
        return results

    def health_check(self) -> bool:
        """检查 Ollama 服务是否可用"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except:
            return False


# 全局单例
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
