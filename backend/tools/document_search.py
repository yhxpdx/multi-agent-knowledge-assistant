"""
文档搜索工具 — 从 Milvus 检索相关文档片段

设计决策：
1. 为什么封装为工具？
   - Agent 通过 Function Calling 调用，需要标准化的工具接口
   - 返回格式化的结果，便于 LLM 理解和引用

2. 返回格式设计：
   - 包含来源引用（文件名 + chunk 编号）
   - 包含相似度分数，便于 LLM 判断相关性
   - 限制返回长度，避免占用过多 context
"""

from langchain_core.tools import tool
from backend.core.milvus_client import get_milvus_manager
from backend.core.embedding import get_embedding_service


@tool
def document_search(query: str, top_k: int = 5) -> str:
    """从知识库中搜索与查询相关的文档片段。

    Args:
        query: 搜索查询文本
        top_k: 返回的结果数量，默认 5
    """
    embedding_service = get_embedding_service()
    milvus = get_milvus_manager()

    # 生成查询向量
    query_embedding = embedding_service.embed_query(query)
    if not query_embedding:
        return "无法生成查询向量，请检查 Embedding 服务。"

    # 搜索
    results = milvus.search(query_embedding, top_k=top_k)
    if not results:
        return "未找到相关文档片段。"

    # 格式化结果
    formatted = []
    for i, r in enumerate(results, 1):
        source = r["metadata"].get("source", "unknown")
        score = r["score"]
        content = r["content"][:500]  # 限制长度
        formatted.append(f"[{i}] 来源: {source} | 相似度: {score:.3f}\n{content}")

    return "\n\n".join(formatted)
