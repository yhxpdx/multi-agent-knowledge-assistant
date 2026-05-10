"""
Embedding 模型评估脚本

评估目标：用实际知识库数据对比 nomic-embed-text 和 bge-m3 的检索效果
评估方法：构建测试集，计算 Recall@K 和 MRR 指标

全链路思考 — 为什么这样评估：
1. 用实际数据而非通用 benchmark：项目数据以中文技术文档为主，通用 benchmark 可能无法反映真实效果
2. Recall@K 是 RAG 最核心指标：用户问一个问题，Top-K 结果中是否包含正确答案
3. MRR 考虑排名位置：正确答案排在第 1 位比排在第 5 位好
4. 同时测试中英文查询：项目可能处理中英文混合输入
"""

import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OLLAMA_URL = "http://localhost:11434/api/embeddings"


def load_documents() -> list[dict]:
    """加载所有知识库文档"""
    docs = []
    for json_file in DATA_DIR.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            docs.extend(json.load(f))
    return docs


def get_embedding(text: str, model: str) -> Optional[list[float]]:
    """获取文本的 embedding 向量"""
    try:
        resp = requests.post(OLLAMA_URL, json={"model": model, "prompt": text}, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("embedding", [])
    except Exception as e:
        print(f"  Error getting embedding: {e}")
    return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算余弦相似度"""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_test_set(docs: list[dict]) -> list[dict]:
    """
    构建测试集

    测试集设计思路：
    - 从每篇文档的标题和内容中提取关键词，构建自然语言查询
    - 每个查询关联 1-2 篇应该被检索到的文档
    - 包含中英文查询，测试多语言能力
    """
    test_cases = [
        # 中文查询 — 测试中文检索能力
        {
            "query": "什么是 RAG 技术？它的核心流程是什么？",
            "expected_titles": ["RAG 检索增强生成技术详解"],
            "lang": "zh",
        },
        {
            "query": "如何设计一个多智能体系统？有哪些常见的协作模式？",
            "expected_titles": ["AI Agent 核心架构与设计模式", "LangGraph 多智能体编排实战"],
            "lang": "zh",
        },
        {
            "query": "向量数据库 Milvus 和 ChromaDB 有什么区别？",
            "expected_titles": ["向量数据库技术选型与实践"],
            "lang": "zh",
        },
        {
            "query": "Embedding 模型怎么选？评估指标有哪些？",
            "expected_titles": ["Embedding 模型选型与评估方法"],
            "lang": "zh",
        },
        {
            "query": "如何用 LangGraph 实现 Supervisor 模式？",
            "expected_titles": ["LangGraph 多智能体编排实战"],
            "lang": "zh",
        },
        {
            "query": "Function Calling 的工作原理是什么？",
            "expected_titles": ["Function Calling 与工具设计"],
            "lang": "zh",
        },
        {
            "query": "Redis 在 AI 应用中怎么用？",
            "expected_titles": ["Redis 在 AI 应用中的最佳实践"],
            "lang": "zh",
        },
        {
            "query": "FastAPI 如何实现流式响应？",
            "expected_titles": ["FastAPI 构建 LLM 应用后端"],
            "lang": "zh",
        },
        {
            "query": "Prompt Engineering 有哪些最佳实践？",
            "expected_titles": ["Prompt Engineering 最佳实践"],
            "lang": "zh",
        },
        {
            "query": "Docker 部署 AI 应用有哪些注意事项？",
            "expected_titles": ["Docker 容器化 AI 应用部署"],
            "lang": "zh",
        },
        # 英文查询 — 测试英文检索能力
        {
            "query": "How does RAG retrieval augmented generation work?",
            "expected_titles": ["RAG 检索增强生成技术详解"],
            "lang": "en",
        },
        {
            "query": "What are the key components of an AI Agent?",
            "expected_titles": ["AI Agent 核心架构与设计模式"],
            "lang": "en",
        },
        {
            "query": "How to choose a vector database for production?",
            "expected_titles": ["向量数据库技术选型与实践"],
            "lang": "en",
        },
        {
            "query": "What is LangGraph and how to use it for multi-agent orchestration?",
            "expected_titles": ["LangGraph 多智能体编排实战"],
            "lang": "en",
        },
        {
            "query": "How to design tools for LLM function calling?",
            "expected_titles": ["Function Calling 与工具设计"],
            "lang": "en",
        },
        # LangChain 文档查询
        {
            "query": "LangChain 中的 Agent 是如何工作的？",
            "expected_titles": ["Agents"],
            "lang": "zh",
        },
        {
            "query": "How to use LangChain retrievers?",
            "expected_titles": ["Retrievers"],
            "lang": "en",
        },
        {
            "query": "LangChain 文本分割器有哪些类型？",
            "expected_titles": ["Text_Splitters"],
            "lang": "zh",
        },
    ]

    # 为每个测试用例找到对应的文档索引
    title_to_idx = {doc["title"]: i for i, doc in enumerate(docs)}
    for tc in test_cases:
        tc["expected_indices"] = [
            title_to_idx[t] for t in tc["expected_titles"] if t in title_to_idx
        ]

    return test_cases


def evaluate_model(
    model_name: str,
    docs: list[dict],
    test_cases: list[dict],
    k: int = 5,
) -> dict:
    """
    评估单个 Embedding 模型

    评估流程：
    1. 为所有文档计算 embedding
    2. 为每个测试查询计算 embedding
    3. 计算查询与所有文档的余弦相似度
    4. 取 Top-K 结果，检查是否包含预期文档
    5. 计算 Recall@K 和 MRR
    """
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"{'='*60}")

    # Step 1: 计算所有文档的 embedding
    print(f"\n[1/3] 计算 {len(docs)} 篇文档的 embedding...")
    doc_embeddings = []
    total_time = 0
    for i, doc in enumerate(docs):
        # 使用标题 + 内容前500字作为文档表示
        text = f"{doc['title']}: {doc['content'][:500]}"
        start = time.time()
        emb = get_embedding(text, model_name)
        elapsed = time.time() - start
        total_time += elapsed
        if emb:
            doc_embeddings.append(emb)
        else:
            print(f"  [FAIL] Failed to embed doc {i}: {doc['title']}")
            doc_embeddings.append(None)
        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(docs)} docs")

    valid_embeddings = [e for e in doc_embeddings if e is not None]
    avg_doc_time = total_time / len(docs) if docs else 0
    print(f"  完成！平均耗时: {avg_doc_time:.3f}s/doc")
    print(f"  向量维度: {len(valid_embeddings[0]) if valid_embeddings else 'N/A'}")

    # Step 2: 评估每个测试查询
    print(f"\n[2/3] 评估 {len(test_cases)} 个测试查询...")
    results = {
        "model": model_name,
        "embedding_dim": len(valid_embeddings[0]) if valid_embeddings else 0,
        "total_docs": len(docs),
        "total_queries": len(test_cases),
        "k": k,
        "recall_at_k": 0,
        "mrr": 0,
        "avg_query_time": 0,
        "details": [],
    }

    recall_hits = 0
    mrr_sum = 0
    query_times = []

    for tc in test_cases:
        start = time.time()
        query_emb = get_embedding(tc["query"], model_name)
        query_time = time.time() - start
        query_times.append(query_time)

        if not query_emb:
            results["details"].append({
                "query": tc["query"],
                "status": "embedding_failed",
            })
            continue

        # 计算与所有文档的相似度
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            if doc_emb is not None:
                sim = cosine_similarity(query_emb, doc_emb)
                similarities.append((sim, i))

        # 排序取 Top-K
        similarities.sort(reverse=True)
        top_k_indices = [idx for _, idx in similarities[:k]]

        # 检查是否命中
        expected = set(tc["expected_indices"])
        hits = expected & set(top_k_indices)
        if hits:
            recall_hits += 1
            # 计算 MRR（第一个命中结果的排名倒数）
            for rank, idx in enumerate(top_k_indices, 1):
                if idx in expected:
                    mrr_sum += 1.0 / rank
                    break

        results["details"].append({
            "query": tc["query"],
            "lang": tc["lang"],
            "expected": tc["expected_titles"],
            "top_k": [docs[idx]["title"] for idx in top_k_indices],
            "hit": len(hits) > 0,
            "first_hit_rank": next(
                (rank for rank, idx in enumerate(top_k_indices, 1) if idx in expected), None
            ),
        })

    # Step 3: 计算汇总指标
    results["recall_at_k"] = recall_hits / len(test_cases) if test_cases else 0
    results["mrr"] = mrr_sum / len(test_cases) if test_cases else 0
    results["avg_query_time"] = sum(query_times) / len(query_times) if query_times else 0

    # 分语言统计
    zh_cases = [tc for tc in test_cases if tc["lang"] == "zh"]
    en_cases = [tc for tc in test_cases if tc["lang"] == "en"]
    zh_hits = sum(1 for d in results["details"] if d.get("lang") == "zh" and d.get("hit"))
    en_hits = sum(1 for d in results["details"] if d.get("lang") == "en" and d.get("hit"))
    results["recall_zh"] = zh_hits / len(zh_cases) if zh_cases else 0
    results["recall_en"] = en_hits / len(en_cases) if en_cases else 0

    print(f"\n[3/3] 评估结果:")
    print(f"  Recall@{k}:    {results['recall_at_k']:.2%}")
    print(f"  Recall@{k} (中文): {results['recall_zh']:.2%}")
    print(f"  Recall@{k} (英文): {results['recall_en']:.2%}")
    print(f"  MRR:         {results['mrr']:.2%}")
    print(f"  平均查询耗时: {results['avg_query_time']:.3f}s")

    return results


def print_comparison(nomic_results: dict, bge_results: dict):
    """打印对比报告"""
    print(f"\n{'='*60}")
    print("Embedding 模型对比报告")
    print(f"{'='*60}")
    print(f"\n{'指标':<25} {'nomic-embed-text':<20} {'bge-m3':<20}")
    print("-" * 65)
    print(f"{'向量维度':<25} {nomic_results['embedding_dim']:<20} {bge_results['embedding_dim']:<20}")
    print(f"{'Recall@5':<25} {nomic_results['recall_at_k']:<20.2%} {bge_results['recall_at_k']:<20.2%}")
    print(f"{'Recall@5 (中文)':<25} {nomic_results['recall_zh']:<20.2%} {bge_results['recall_zh']:<20.2%}")
    print(f"{'Recall@5 (英文)':<25} {nomic_results['recall_en']:<20.2%} {bge_results['recall_en']:<20.2%}")
    print(f"{'MRR':<25} {nomic_results['mrr']:<20.2%} {bge_results['mrr']:<20.2%}")
    print(f"{'平均查询耗时(s)':<25} {nomic_results['avg_query_time']:<20.3f} {bge_results['avg_query_time']:<20.3f}")

    # 给出推荐
    print(f"\n{'='*60}")
    print("选型结论")
    print(f"{'='*60}")

    if bge_results["recall_at_k"] > nomic_results["recall_at_k"]:
        winner = "bge-m3"
        reason = f"检索准确率更高 (Recall@5: {bge_results['recall_at_k']:.2%} vs {nomic_results['recall_at_k']:.2%})"
    elif nomic_results["recall_at_k"] > bge_results["recall_at_k"]:
        winner = "nomic-embed-text"
        reason = f"检索准确率更高 (Recall@5: {nomic_results['recall_at_k']:.2%} vs {bge_results['recall_at_k']:.2%})"
    else:
        if nomic_results["avg_query_time"] < bge_results["avg_query_time"]:
            winner = "nomic-embed-text"
            reason = "准确率相当，但查询速度更快"
        else:
            winner = "bge-m3"
            reason = "准确率相当，中文效果更好"

    print(f"\n推荐选择: {winner}")
    print(f"理由: {reason}")

    # 保存报告
    report = {
        "nomic": nomic_results,
        "bge": bge_results,
        "recommendation": winner,
        "reason": reason,
    }
    report_path = Path(__file__).parent / "embedding_eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告已保存到: {report_path}")


def main():
    # 加载文档
    docs = load_documents()
    print(f"加载了 {len(docs)} 篇文档")

    # 构建测试集
    test_cases = build_test_set(docs)
    print(f"构建了 {len(test_cases)} 个测试用例")

    # 评估 nomic-embed-text
    nomic_results = evaluate_model("nomic-embed-text", docs, test_cases)

    # 评估 bge-m3
    bge_results = evaluate_model("bge-m3", docs, test_cases)

    # 对比报告
    print_comparison(nomic_results, bge_results)


if __name__ == "__main__":
    main()
