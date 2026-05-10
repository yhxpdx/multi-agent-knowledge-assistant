"""
数据导入脚本 — 将采集的知识库数据导入 Milvus

流程：
1. 加载 JSON 数据文件
2. 对每篇文档进行分块
3. 生成 Embedding 向量
4. 存储到 Milvus
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from backend.core.document_parser import get_document_parser
from backend.core.embedding import get_embedding_service
from backend.core.milvus_client import get_milvus_manager

DATA_DIR = Path(__file__).parent / "raw"


def ingest():
    print("=" * 60)
    print("数据导入开始")
    print("=" * 60)

    # 初始化服务
    parser = get_document_parser()
    embedding_service = get_embedding_service()
    milvus = get_milvus_manager()

    # 创建集合
    print("\n[1/4] 创建 Milvus 集合...")
    milvus.create_collection()

    # 加载所有 JSON 数据
    all_docs = []
    for json_file in DATA_DIR.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            docs = json.load(f)
            all_docs.extend(docs)
            print(f"  加载 {json_file.name}: {len(docs)} 篇文档")

    print(f"\n[2/4] 共 {len(all_docs)} 篇文档待导入")

    # 逐篇处理
    total_chunks = 0
    for i, doc in enumerate(all_docs):
        title = doc.get("title", "unknown")
        content = doc.get("content", "")

        if len(content) < 50:
            print(f"  [{i+1}/{len(all_docs)}] 跳过过短文档: {title}")
            continue

        # 分块
        chunks = parser.text_splitter.split_text(content)
        if not chunks:
            continue

        # 生成 embedding
        embeddings = embedding_service.embed_documents(chunks)
        if not all(embeddings):
            print(f"  [{i+1}/{len(all_docs)}] Embedding 失败: {title}")
            continue

        # 构建元数据
        metadatas = [
            {
                "source": doc.get("source", "unknown"),
                "title": title,
                "chunk_index": j,
                "total_chunks": len(chunks),
            }
            for j in range(len(chunks))
        ]

        # 存储到 Milvus
        doc_id = f"doc_{i:04d}"
        milvus.insert(doc_id, chunks, embeddings, metadatas)
        total_chunks += len(chunks)

        print(f"  [{i+1}/{len(all_docs)}] {title}: {len(chunks)} chunks")

    # 加载集合到内存
    print("\n[3/4] 加载集合到内存...")
    milvus.load_collection()

    # 统计
    stats = milvus.get_stats()
    print(f"\n{'='*60}")
    print(f"导入完成！")
    print(f"  文档数: {len(all_docs)}")
    print(f"  总 chunks: {total_chunks}")
    print(f"  Milvus 集合: {stats['collection']}")
    print(f"  Milvus 记录数: {stats['count']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    ingest()
