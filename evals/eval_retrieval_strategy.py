"""
检索策略评估 — 任务 3.4

对比纯向量搜索 vs 混合搜索（向量 + 关键词）

评估维度：
1. 检索准确率（Recall@5, MRR）
2. 排序合理性（相关文档排名）
3. 不同查询类型的表现差异

评估方法：
- 使用已导入 Milvus 的知识库数据
- 设计测试查询集，覆盖不同查询类型
- 对比两种策略的检索结果
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.embedding import get_embedding_service
from backend.core.milvus_client import get_milvus_manager


# 测试查询集 — 设计原则：
# 1. 精确匹配：查询直接包含文档中的关键词
# 2. 语义匹配：查询与文档语义相同但用词不同
# 3. 多概念查询：查询包含多个概念，需要综合检索
TEST_QUERIES = [
    {
        "query": "什么是 RAG 技术？",
        "expected_keywords": ["RAG", "检索", "生成"],
        "category": "精确匹配",
    },
    {
        "query": "向量数据库如何存储和检索文档？",
        "expected_keywords": ["Milvus", "向量", "检索", "索引"],
        "category": "语义匹配",
    },
    {
        "query": "LangGraph 的 Supervisor 模式是什么？",
        "expected_keywords": ["LangGraph", "Supervisor", "路由", "Agent"],
        "category": "精确匹配",
    },
    {
        "query": "如何实现多智能体协作？",
        "expected_keywords": ["Agent", "协作", "路由", "编排"],
        "category": "语义匹配",
    },
    {
        "query": "Embedding 模型怎么选？",
        "expected_keywords": ["Embedding", "bge-m3", "模型", "评估"],
        "category": "语义匹配",
    },
    {
        "query": "文档分块策略有哪些？",
        "expected_keywords": ["分块", "chunk", "overlap", "策略"],
        "category": "精确匹配",
    },
    {
        "query": "Redis 在对话系统中的作用",
        "expected_keywords": ["Redis", "对话", "记忆", "历史"],
        "category": "语义匹配",
    },
    {
        "query": "Function Calling 工具调用",
        "expected_keywords": ["工具", "Function", "调用", "Agent"],
        "category": "精确匹配",
    },
    {
        "query": "HNSW 索引的原理和优势",
        "expected_keywords": ["HNSW", "索引", "向量", "搜索"],
        "category": "精确匹配",
    },
    {
        "query": "AI Agent 的核心模块有哪些？",
        "expected_keywords": ["Agent", "规划", "记忆", "工具"],
        "category": "语义匹配",
    },
    {
        "query": "如何评估检索效果？",
        "expected_keywords": ["Recall", "MRR", "评估", "检索"],
        "category": "语义匹配",
    },
    {
        "query": "SSE 流式输出怎么实现？",
        "expected_keywords": ["SSE", "流式", "FastAPI", "输出"],
        "category": "精确匹配",
    },
    {
        "query": "Docker 部署向量数据库",
        "expected_keywords": ["Docker", "Milvus", "部署", "容器"],
        "category": "多概念查询",
    },
    {
        "query": "如何处理中文文档的检索？",
        "expected_keywords": ["中文", "分词", "Embedding", "检索"],
        "category": "语义匹配",
    },
    {
        "query": "LLM 幻觉问题怎么解决？",
        "expected_keywords": ["幻觉", "RAG", "检索", "上下文"],
        "category": "语义匹配",
    },
]


def vector_search_only(query: str, embedding_service, milvus_manager, top_k: int = 5) -> list[dict]:
    """纯向量搜索"""
    query_embedding = embedding_service.embed_query(query)
    if not query_embedding:
        return []
    return milvus_manager.search(query_embedding, top_k=top_k)


def hybrid_search(query: str, embedding_service, milvus_manager, top_k: int = 5) -> list[dict]:
    """混合搜索（向量 + 关键词加权）

    策略：
    1. 向量搜索获取 Top-K*2 结果
    2. 对结果进行关键词匹配打分
    3. 综合向量分数和关键词分数重新排序
    """
    query_embedding = embedding_service.embed_query(query)
    if not query_embedding:
        return []

    # 获取更多候选结果
    candidates = milvus_manager.search(query_embedding, top_k=top_k * 2)

    # 提取查询关键词
    import jieba
    query_keywords = set(jieba.cut(query))
    query_keywords = {w for w in query_keywords if len(w) > 1}  # 过滤单字

    # 计算综合分数
    for doc in candidates:
        content = doc.get("content", "")
        # 关键词匹配分数
        keyword_hits = sum(1 for kw in query_keywords if kw in content)
        keyword_score = keyword_hits / max(len(query_keywords), 1)

        # 综合分数：向量分数 * 0.7 + 关键词分数 * 0.3
        vector_score = doc.get("score", 0)
        doc["hybrid_score"] = vector_score * 0.7 + keyword_score * 0.3
        doc["keyword_hits"] = keyword_hits

    # 按综合分数排序
    candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return candidates[:top_k]


def evaluate_strategy(results: list[dict], expected_keywords: list[str]) -> dict:
    """评估单个查询的检索结果"""
    if not results:
        return {"recall": 0, "mrr": 0, "hits": 0, "total_keywords": len(expected_keywords)}

    # 检查关键词命中
    keyword_hits = 0
    first_relevant_rank = None

    for rank, doc in enumerate(results):
        content = doc.get("content", "")
        doc_hits = sum(1 for kw in expected_keywords if kw in content)
        if doc_hits > 0:
            keyword_hits = max(keyword_hits, doc_hits)
            if first_relevant_rank is None:
                first_relevant_rank = rank + 1

    recall = 1.0 if keyword_hits > 0 else 0.0
    mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    return {
        "recall": recall,
        "mrr": mrr,
        "hits": keyword_hits,
        "total_keywords": len(expected_keywords),
        "first_relevant_rank": first_relevant_rank,
    }


def run_evaluation():
    """运行完整评估"""
    print("=" * 70)
    print("检索策略评估 — 纯向量搜索 vs 混合搜索")
    print("=" * 70)

    embedding_service = get_embedding_service()
    milvus_manager = get_milvus_manager()
    milvus_manager.create_collection()
    milvus_manager.load_collection()

    # 检查服务状态
    if not embedding_service.health_check():
        print("错误：Ollama 服务不可用")
        return

    print(f"\n测试查询数量: {len(TEST_QUERIES)}")
    print("-" * 70)

    vector_results = []
    hybrid_results = []

    for i, tq in enumerate(TEST_QUERIES):
        query = tq["query"]
        expected = tq["expected_keywords"]
        category = tq["category"]

        print(f"\n[{i+1}/{len(TEST_QUERIES)}] {query}")
        print(f"  类别: {category}")
        print(f"  期望关键词: {expected}")

        # 纯向量搜索
        v_docs = vector_search_only(query, embedding_service, milvus_manager)
        v_eval = evaluate_strategy(v_docs, expected)
        vector_results.append(v_eval)

        # 混合搜索
        h_docs = hybrid_search(query, embedding_service, milvus_manager)
        h_eval = evaluate_strategy(h_docs, expected)
        hybrid_results.append(h_eval)

        print(f"  向量搜索: Recall={v_eval['recall']:.2f}, MRR={v_eval['mrr']:.2f}, "
              f"命中={v_eval['hits']}/{v_eval['total_keywords']}")
        print(f"  混合搜索: Recall={h_eval['recall']:.2f}, MRR={h_eval['mrr']:.2f}, "
              f"命中={h_eval['hits']}/{h_eval['total_keywords']}")

    # 汇总
    print("\n" + "=" * 70)
    print("评估汇总")
    print("=" * 70)

    v_avg_recall = sum(r["recall"] for r in vector_results) / len(vector_results)
    v_avg_mrr = sum(r["mrr"] for r in vector_results) / len(vector_results)
    h_avg_recall = sum(r["recall"] for r in hybrid_results) / len(hybrid_results)
    h_avg_mrr = sum(r["mrr"] for r in hybrid_results) / len(hybrid_results)

    print(f"\n{'策略':<15} {'Recall@5':<12} {'MRR':<12}")
    print("-" * 40)
    print(f"{'纯向量搜索':<15} {v_avg_recall:<12.2%} {v_avg_mrr:<12.2%}")
    print(f"{'混合搜索':<15} {h_avg_recall:<12.2%} {h_avg_mrr:<12.2%}")

    # 按类别分析
    categories = {}
    for tq, v, h in zip(TEST_QUERIES, vector_results, hybrid_results):
        cat = tq["category"]
        if cat not in categories:
            categories[cat] = {"vector": [], "hybrid": []}
        categories[cat]["vector"].append(v)
        categories[cat]["hybrid"].append(h)

    print(f"\n{'类别':<15} {'向量Recall':<12} {'混合Recall':<12} {'差异':<8}")
    print("-" * 50)
    for cat, data in categories.items():
        v_r = sum(r["recall"] for r in data["vector"]) / len(data["vector"])
        h_r = sum(r["recall"] for r in data["hybrid"]) / len(data["hybrid"])
        diff = h_r - v_r
        sign = "+" if diff > 0 else ""
        print(f"{cat:<15} {v_r:<12.2%} {h_r:<12.2%} {sign}{diff:.2%}")

    # 结论
    print("\n" + "=" * 70)
    print("评估结论")
    print("=" * 70)

    if h_avg_recall > v_avg_recall:
        print("混合搜索在检索准确率上优于纯向量搜索。")
        print(f"  Recall 提升: {h_avg_recall - v_avg_recall:.2%}")
    elif h_avg_recall == v_avg_recall:
        print("两种策略在检索准确率上表现相同。")
    else:
        print("纯向量搜索在检索准确率上优于混合搜索。")

    if h_avg_mrr > v_avg_mrr:
        print(f"  MRR 提升: {h_avg_mrr - v_avg_mrr:.2%}")

    print("\n选择建议：")
    if h_avg_recall - v_avg_recall > 0.05:
        print("  推荐使用混合搜索 — 在当前数据集上有明显优势。")
    elif v_avg_recall >= 0.9:
        print("  推荐使用纯向量搜索 — bge-m3 的向量检索已经足够优秀，")
        print("  混合搜索的额外复杂度不值得。")
    else:
        print("  两种策略差异不大，建议使用纯向量搜索以保持架构简洁。")

    return {
        "vector": {"recall": v_avg_recall, "mrr": v_avg_mrr},
        "hybrid": {"recall": h_avg_recall, "mrr": h_avg_mrr},
    }


if __name__ == "__main__":
    run_evaluation()
