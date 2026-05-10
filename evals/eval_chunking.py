"""
分块策略评估脚本

评估目标：找到最优的 chunk_size 和 chunk_overlap 参数组合
评估方法：用不同参数分块后，测试检索召回率

全链路思考 — 为什么这个评估重要：
1. chunk_size 太小：语义不完整，检索到的片段缺乏上下文
2. chunk_size 太大：包含过多无关信息，稀释了关键内容的权重
3. overlap 太小：相邻 chunk 语义断裂，跨 chunk 的问题难以回答
4. overlap 太大：存储浪费，检索结果冗余
5. 最优参数取决于数据特点：技术文档通常段落较长，需要实验确定
"""

import json
import time
import sys
import os
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "bge-m3"


def load_documents() -> list[dict]:
    docs = []
    for json_file in DATA_DIR.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            docs.extend(json.load(f))
    return docs


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """手动实现文本分块，用于评估不同参数"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # 尝试在句子边界分割
        if end < len(text):
            # 在 chunk 末尾找句号、换行等分隔点
            for sep in ["\n\n", "\n", "。", ".", "！", "!", "？", "?"]:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size * 0.5:  # 至少保留一半内容
                    end = start + last_sep + len(sep)
                    chunk = text[start:end]
                    break

        chunks.append(chunk.strip())
        start = end - chunk_overlap

    return [c for c in chunks if len(c) > 10]  # 过滤太短的块


def get_embedding(text: str) -> list[float]:
    resp = requests.post(OLLAMA_URL, json={"model": EMBEDDING_MODEL, "prompt": text}, timeout=30)
    if resp.status_code == 200:
        return resp.json().get("embedding", [])
    return []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def evaluate_chunking_params(
    docs: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    test_queries: list[dict],
    k: int = 5,
) -> dict:
    """评估单组分块参数"""
    # Step 1: 分块
    all_chunks = []
    chunk_to_doc = {}  # chunk index -> doc title
    for doc in docs:
        chunks = chunk_text(doc["content"], chunk_size, chunk_overlap)
        for chunk in chunks:
            chunk_to_doc[len(all_chunks)] = doc["title"]
            all_chunks.append(chunk)

    # Step 2: 计算所有 chunk 的 embedding
    chunk_embeddings = []
    for chunk in all_chunks:
        emb = get_embedding(chunk[:500])  # 限制长度避免太慢
        chunk_embeddings.append(emb)
        time.sleep(0.05)  # 避免过快

    # Step 3: 评估每个查询
    recall_hits = 0
    mrr_sum = 0

    for tc in test_queries:
        query_emb = get_embedding(tc["query"])
        if not query_emb:
            continue

        # 计算相似度
        similarities = []
        for i, chunk_emb in enumerate(chunk_embeddings):
            if chunk_emb:
                sim = cosine_similarity(query_emb, chunk_emb)
                similarities.append((sim, i))

        similarities.sort(reverse=True)
        top_k_indices = [idx for _, idx in similarities[:k]]

        # 检查是否命中预期文档
        expected_titles = set(tc["expected_titles"])
        hit = any(chunk_to_doc.get(idx) in expected_titles for idx in top_k_indices)
        if hit:
            recall_hits += 1
            for rank, idx in enumerate(top_k_indices, 1):
                if chunk_to_doc.get(idx) in expected_titles:
                    mrr_sum += 1.0 / rank
                    break

    n = len(test_queries)
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "total_chunks": len(all_chunks),
        "avg_chunk_len": sum(len(c) for c in all_chunks) / len(all_chunks) if all_chunks else 0,
        "recall_at_k": recall_hits / n if n else 0,
        "mrr": mrr_sum / n if n else 0,
    }


def main():
    docs = load_documents()
    print(f"加载了 {len(docs)} 篇文档")

    # 测试查询集
    test_queries = [
        {"query": "什么是 RAG 技术？", "expected_titles": ["RAG 检索增强生成技术详解"]},
        {"query": "如何设计多智能体系统？", "expected_titles": ["AI Agent 核心架构与设计模式", "LangGraph 多智能体编排实战"]},
        {"query": "Milvus 和 ChromaDB 的区别", "expected_titles": ["向量数据库技术选型与实践"]},
        {"query": "Embedding 模型怎么选？", "expected_titles": ["Embedding 模型选型与评估方法"]},
        {"query": "LangGraph Supervisor 模式", "expected_titles": ["LangGraph 多智能体编排实战"]},
        {"query": "Function Calling 工作原理", "expected_titles": ["Function Calling 与工具设计"]},
        {"query": "Redis 在 AI 应用中的使用", "expected_titles": ["Redis 在 AI 应用中的最佳实践"]},
        {"query": "FastAPI 流式响应", "expected_titles": ["FastAPI 构建 LLM 应用后端"]},
        {"query": "Prompt Engineering 最佳实践", "expected_titles": ["Prompt Engineering 最佳实践"]},
        {"query": "Docker 部署 AI 应用", "expected_titles": ["Docker 容器化 AI 应用部署"]},
    ]

    # 参数组合
    param_combos = [
        (200, 20),   # 小块，小重叠
        (200, 50),   # 小块，中重叠
        (500, 50),   # 中块，中重叠（当前默认）
        (500, 100),  # 中块，大重叠
        (800, 80),   # 大块，中重叠
        (800, 150),  # 大块，大重叠
        (1000, 100), # 大块，大重叠
    ]

    print(f"\n评估 {len(param_combos)} 种参数组合...")
    results = []

    for chunk_size, chunk_overlap in param_combos:
        print(f"\n  评估: chunk_size={chunk_size}, overlap={chunk_overlap}")
        result = evaluate_chunking_params(docs, chunk_size, chunk_overlap, test_queries)
        results.append(result)
        print(f"    chunks: {result['total_chunks']}, avg_len: {result['avg_chunk_len']:.0f}")
        print(f"    Recall@5: {result['recall_at_k']:.2%}, MRR: {result['mrr']:.2%}")

    # 打印对比表
    print(f"\n{'='*80}")
    print("分块策略对比报告")
    print(f"{'='*80}")
    print(f"\n{'chunk_size':<12} {'overlap':<10} {'chunks':<10} {'avg_len':<10} {'Recall@5':<12} {'MRR':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['chunk_size']:<12} {r['chunk_overlap']:<10} {r['total_chunks']:<10} {r['avg_chunk_len']:<10.0f} {r['recall_at_k']:<12.2%} {r['mrr']:<10.2%}")

    # 找最优
    best = max(results, key=lambda x: (x["recall_at_k"], x["mrr"]))
    print(f"\n最优参数: chunk_size={best['chunk_size']}, overlap={best['chunk_overlap']}")
    print(f"  Recall@5: {best['recall_at_k']:.2%}, MRR: {best['mrr']:.2%}")
    print(f"  总 chunks: {best['total_chunks']}, 平均长度: {best['avg_chunk_len']:.0f}")

    # 保存报告
    report_path = Path(__file__).parent / "chunking_eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best}, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
