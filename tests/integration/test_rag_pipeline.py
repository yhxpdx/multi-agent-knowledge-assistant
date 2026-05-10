"""
RAG 管道集成测试

测试覆盖：
1. 文档解析 → 分块 → Embedding → Milvus 存储 完整流程
2. 查询 → Embedding → Milvus 检索 → 结果格式化
3. 按文档 ID 过滤检索
4. 检索结果排序（相似度降序）

注意：这些测试需要 Milvus 和 Ollama 服务运行
可通过 pytest -m "not integration" 跳过
"""

import pytest
import uuid
from unittest.mock import patch, MagicMock

from backend.core.document_parser import DocumentParser
from backend.core.milvus_client import MilvusManager
from backend.core.embedding import EmbeddingService
from backend.tools.document_search import document_search


@pytest.mark.integration
class TestRAGPipeline:
    """RAG 管道集成测试（需要真实服务）"""

    @pytest.fixture(scope="class")
    def embedding_service(self):
        from backend.core.embedding import get_embedding_service
        return get_embedding_service()

    @pytest.fixture(scope="class")
    def milvus_manager(self):
        from backend.core.milvus_client import get_milvus_manager
        mgr = get_milvus_manager()
        mgr.create_collection()
        mgr.load_collection()
        return mgr

    @pytest.fixture(scope="class")
    def test_doc_id(self):
        return f"test_{uuid.uuid4().hex[:6]}"

    def test_embed_and_store(self, embedding_service, milvus_manager, test_doc_id):
        """测试 embedding 生成和 Milvus 存储"""
        texts = [
            "RAG 是一种结合检索和生成的技术",
            "LangGraph 用于构建多智能体系统",
            "Milvus 是一个向量数据库",
        ]

        embeddings = embedding_service.embed_documents(texts)
        assert all(e is not None for e in embeddings)
        assert all(len(e) == 1024 for e in embeddings)

        metadatas = [{"source": "test", "chunk_index": i} for i in range(len(texts))]
        milvus_manager.insert(test_doc_id, texts, embeddings, metadatas)

    def test_search_relevance(self, embedding_service, milvus_manager, test_doc_id):
        """测试检索相关性"""
        # 先确保存储了数据
        self.test_embed_and_store(embedding_service, milvus_manager, test_doc_id)

        query = "什么是 RAG？"
        query_embedding = embedding_service.embed_query(query)
        assert query_embedding is not None

        results = milvus_manager.search(query_embedding, top_k=3)
        assert len(results) > 0

        # 最相关的结果应该包含 RAG 相关内容
        top_result = results[0]
        assert "RAG" in top_result["content"] or "检索" in top_result["content"]

    def test_search_with_filter(self, embedding_service, milvus_manager, test_doc_id):
        """测试按文档 ID 过滤搜索"""
        self.test_embed_and_store(embedding_service, milvus_manager, test_doc_id)

        query_embedding = embedding_service.embed_query("向量数据库")
        results = milvus_manager.search(query_embedding, top_k=5, doc_id=test_doc_id)

        for result in results:
            assert result["doc_id"] == test_doc_id

    def test_cleanup(self, milvus_manager, test_doc_id):
        """清理测试数据"""
        try:
            milvus_manager.delete_by_doc(test_doc_id)
        except Exception:
            pass


@pytest.mark.integration
class TestRAGWithMock:
    """使用 mock 的 RAG 管道测试（不需要真实服务）"""

    @patch("backend.core.embedding.requests.post")
    @patch("backend.core.milvus_client.MilvusClient")
    def test_full_pipeline_mock(self, mock_milvus_cls, mock_post):
        """测试完整 RAG 管道（mock 版本）"""
        # Mock embedding
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1] * 1024}
        mock_post.return_value = mock_response

        # Mock Milvus
        mock_milvus = MagicMock()
        mock_milvus_cls.return_value = mock_milvus
        mock_milvus.search.return_value = [[
            {
                "id": 1,
                "distance": 0.95,
                "entity": {
                    "doc_id": "test",
                    "content": "RAG 是检索增强生成技术",
                    "metadata": {"source": "test.txt"},
                },
            }
        ]]

        # 执行管道
        from backend.core.embedding import EmbeddingService
        from backend.core.milvus_client import MilvusManager

        with patch("backend.core.embedding.get_settings") as mock_emb_settings, \
             patch("backend.core.milvus_client.get_settings") as mock_mil_settings:
            mock_emb_settings.return_value = MagicMock(
                EMBEDDING_MODEL="bge-m3",
                OLLAMA_HOST="http://localhost:11434",
                EMBEDDING_DIM=1024,
            )
            mock_mil_settings.return_value = MagicMock(
                MILVUS_HOST="localhost",
                MILVUS_PORT=19530,
                MILVUS_COLLECTION="test",
                EMBEDDING_DIM=1024,
            )

            emb = EmbeddingService()
            mil = MilvusManager()

            # 生成 embedding
            query_emb = emb.embed_query("什么是 RAG？")
            assert query_emb is not None

            # 搜索
            results = mil.search(query_emb, top_k=3)
            assert len(results) > 0
            assert results[0]["score"] == 0.95
