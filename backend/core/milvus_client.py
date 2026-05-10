"""
Milvus 向量数据库客户端

设计决策（全链路思考）：
1. 为什么用 MilvusClient 而不是 ORM API？
   - PyMilvus 3.0 推荐使用 MilvusClient，ORM API 将在 3.1 移除
   - MilvusClient API 更简洁，一行代码完成操作

2. 集合 schema 设计：
   - id: 自增主键
   - doc_id: 文档 ID，用于关联原始文档
   - content: 文本内容
   - embedding: 向量（1024 维，bge-m3）
   - metadata: JSON 格式的元数据（来源、页码等）

3. 索引选择：HNSW
   - 为什么选 HNSW 而不是 IVF_FLAT？
     - 我们的数据规模（几百到几千条）HNSW 更合适
     - HNSW 查询速度快，不需要训练
     - IVF_FLAT 需要 nlist 参数，小数据集反而不好调
   - 参数：M=16, efConstruction=200（默认值，适合中小规模）
"""

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from typing import Optional
from backend.core.config import get_settings


class MilvusManager:
    def __init__(self):
        self.settings = get_settings()
        self.client = MilvusClient(
            uri=f"http://{self.settings.MILVUS_HOST}:{self.settings.MILVUS_PORT}"
        )
        self.collection = self.settings.MILVUS_COLLECTION

    def create_collection(self):
        """创建集合（如果不存在）"""
        if self.client.has_collection(self.collection):
            return

        schema = CollectionSchema(fields=[
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("doc_id", DataType.VARCHAR, max_length=64),
            FieldSchema("content", DataType.VARCHAR, max_length=8192),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=self.settings.EMBEDDING_DIM),
            FieldSchema("metadata", DataType.JSON),
        ])

        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
        )

        # 创建 HNSW 索引
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        self.client.create_index(self.collection, index_params)
        print(f"Created collection '{self.collection}' with HNSW index")

    def load_collection(self):
        """加载集合到内存（搜索前必须调用）"""
        self.client.load_collection(self.collection)

    def insert(self, doc_id: str, contents: list[str], embeddings: list[list[float]], metadatas: list[dict]):
        """插入文档 chunks"""
        data = [
            {"doc_id": doc_id, "content": c, "embedding": e, "metadata": m}
            for c, e, m in zip(contents, embeddings, metadatas)
        ]
        return self.client.insert(collection_name=self.collection, data=data)

    def search(self, query_embedding: list[float], top_k: int = 5, doc_id: Optional[str] = None) -> list[dict]:
        """向量相似度搜索"""
        filter_expr = f'doc_id == "{doc_id}"' if doc_id else None

        results = self.client.search(
            collection_name=self.collection,
            data=[query_embedding],
            limit=top_k,
            output_fields=["doc_id", "content", "metadata"],
            filter=filter_expr,
        )

        # 格式化结果
        formatted = []
        for hit in results[0]:
            formatted.append({
                "id": hit["id"],
                "score": hit["distance"],
                "doc_id": hit["entity"]["doc_id"],
                "content": hit["entity"]["content"],
                "metadata": hit["entity"]["metadata"],
            })
        return formatted

    def delete_by_doc(self, doc_id: str):
        """删除指定文档的所有 chunks"""
        self.client.delete(
            collection_name=self.collection,
            filter=f'doc_id == "{doc_id}"',
        )

    def get_stats(self) -> dict:
        """获取集合统计信息"""
        stats = self.client.get_collection_stats(self.collection)
        return {
            "collection": self.collection,
            "count": stats.get("row_count", 0),
        }

    def drop_collection(self):
        """删除集合（用于测试）"""
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)

    def close(self):
        """关闭连接"""
        self.client.close()


# 全局单例
_milvus_manager: Optional[MilvusManager] = None


def get_milvus_manager() -> MilvusManager:
    global _milvus_manager
    if _milvus_manager is None:
        _milvus_manager = MilvusManager()
    return _milvus_manager
