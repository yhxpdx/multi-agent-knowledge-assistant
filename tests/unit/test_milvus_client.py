"""
Milvus 客户端单元测试

测试覆盖：
1. 集合创建
2. 数据插入
3. 向量搜索
4. 文档删除
5. 集合统计
6. 单例模式
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from backend.core.milvus_client import MilvusManager, get_milvus_manager


class TestMilvusManager:
    """Milvus 客户端测试"""

    @pytest.fixture
    def manager(self):
        """创建管理器实例"""
        with patch("backend.core.milvus_client.get_settings") as mock_settings, \
             patch("backend.core.milvus_client.MilvusClient") as mock_client:
            mock_settings.return_value = MagicMock(
                MILVUS_HOST="localhost",
                MILVUS_PORT=19530,
                MILVUS_COLLECTION="test_collection",
                EMBEDDING_DIM=1024,
            )
            mgr = MilvusManager()
            mgr._mock_client = mock_client
            return mgr

    def test_create_collection_new(self, manager):
        """测试创建新集合"""
        manager.client.has_collection.return_value = False

        manager.create_collection()

        manager.client.create_collection.assert_called_once()
        manager.client.create_index.assert_called_once()

    def test_create_collection_exists(self, manager):
        """测试集合已存在时跳过创建"""
        manager.client.has_collection.return_value = True

        manager.create_collection()

        manager.client.create_collection.assert_not_called()

    def test_load_collection(self, manager):
        """测试加载集合"""
        manager.load_collection()
        manager.client.load_collection.assert_called_once_with("test_collection")

    def test_insert(self, manager):
        """测试数据插入"""
        doc_id = "doc_001"
        contents = ["chunk1", "chunk2"]
        embeddings = [[0.1] * 1024, [0.2] * 1024]
        metadatas = [{"source": "test.txt"}, {"source": "test.txt"}]

        manager.insert(doc_id, contents, embeddings, metadatas)

        manager.client.insert.assert_called_once()
        call_kwargs = manager.client.insert.call_args
        assert call_kwargs[1]["collection_name"] == "test_collection"
        assert len(call_kwargs[1]["data"]) == 2

    def test_search(self, manager):
        """测试向量搜索"""
        mock_results = [[
            {
                "id": 1,
                "distance": 0.95,
                "entity": {
                    "doc_id": "doc_001",
                    "content": "test content",
                    "metadata": {"source": "test.txt"},
                },
            }
        ]]
        manager.client.search.return_value = mock_results

        query_embedding = [0.1] * 1024
        results = manager.search(query_embedding, top_k=5)

        assert len(results) == 1
        assert results[0]["score"] == 0.95
        assert results[0]["content"] == "test content"

    def test_search_with_doc_id_filter(self, manager):
        """测试带文档 ID 过滤的搜索"""
        manager.client.search.return_value = [[]]

        query_embedding = [0.1] * 1024
        manager.search(query_embedding, top_k=5, doc_id="doc_001")

        call_kwargs = manager.client.search.call_args[1]
        assert 'doc_id == "doc_001"' in call_kwargs.get("filter", "")

    def test_delete_by_doc(self, manager):
        """测试按文档 ID 删除"""
        manager.delete_by_doc("doc_001")

        manager.client.delete.assert_called_once()
        call_kwargs = manager.client.delete.call_args[1]
        assert "doc_001" in call_kwargs["filter"]

    def test_get_stats(self, manager):
        """测试获取统计信息"""
        manager.client.get_collection_stats.return_value = {"row_count": 100}

        stats = manager.get_stats()
        assert stats["count"] == 100
        assert stats["collection"] == "test_collection"

    def test_drop_collection(self, manager):
        """测试删除集合"""
        manager.client.has_collection.return_value = True
        manager.drop_collection()
        manager.client.drop_collection.assert_called_once_with("test_collection")

    def test_close(self, manager):
        """测试关闭连接"""
        manager.close()
        manager.client.close.assert_called_once()

    def test_singleton(self):
        """测试单例模式"""
        import backend.core.milvus_client as mod
        mod._milvus_manager = None

        with patch("backend.core.milvus_client.get_settings") as mock_settings, \
             patch("backend.core.milvus_client.MilvusClient"):
            mock_settings.return_value = MagicMock(
                MILVUS_HOST="localhost",
                MILVUS_PORT=19530,
                MILVUS_COLLECTION="test",
                EMBEDDING_DIM=1024,
            )
            m1 = get_milvus_manager()
            m2 = get_milvus_manager()
            assert m1 is m2
