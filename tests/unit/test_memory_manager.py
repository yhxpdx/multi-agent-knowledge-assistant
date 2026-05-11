"""
记忆管理器单元测试

测试覆盖：
1. Collection 创建
2. 记忆添加与去重
3. 记忆检索与格式化
4. 记忆删除
5. 记忆列出与筛选
6. 记忆提取
"""

import pytest
from unittest.mock import patch, MagicMock
import json

from backend.core.memory_manager import MemoryManager, VALID_CATEGORIES


class TestMemoryManager:
    """MemoryManager 测试"""

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    def test_ensure_collection_creates_new(self, mock_settings, mock_emb, mock_milvus):
        """测试创建新 collection"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = False
        mock_milvus.return_value = mock_client

        manager = MemoryManager()
        mock_client.create_collection.assert_called_once()
        mock_client.create_index.assert_called_once()

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    def test_ensure_collection_exists(self, mock_settings, mock_emb, mock_milvus):
        """测试 collection 已存在时不重复创建"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_milvus.return_value = mock_client

        manager = MemoryManager()
        mock_client.create_collection.assert_not_called()

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    def test_add_memory_success(self, mock_settings, mock_emb, mock_milvus):
        """测试成功添加记忆"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client.search.return_value = [[]]
        mock_milvus.return_value = mock_client

        mock_emb_svc = MagicMock()
        mock_emb_svc.embed_query.return_value = [0.1] * 1024
        mock_emb.return_value = mock_emb_svc

        manager = MemoryManager()
        result = manager.add_memory("用户喜欢 Python", "preference", "test_session")

        assert result is True
        mock_client.insert.assert_called_once()

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    def test_add_memory_duplicate(self, mock_settings, mock_emb, mock_milvus):
        """测试相似度 >0.95 时不重复写入"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client.search.return_value = [[{"distance": 0.98, "id": "existing"}]]
        mock_milvus.return_value = mock_client

        mock_emb_svc = MagicMock()
        mock_emb_svc.embed_query.return_value = [0.1] * 1024
        mock_emb.return_value = mock_emb_svc

        manager = MemoryManager()
        result = manager.add_memory("用户喜欢 Python", "preference", "test_session")

        assert result is False
        mock_client.insert.assert_not_called()

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    def test_add_memory_embedding_failure(self, mock_settings, mock_emb, mock_milvus):
        """测试 embedding 失败时不写入"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_milvus.return_value = mock_client

        mock_emb_svc = MagicMock()
        mock_emb_svc.embed_query.return_value = None
        mock_emb.return_value = mock_emb_svc

        manager = MemoryManager()
        result = manager.add_memory("test", "fact", "session")

        assert result is False

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    def test_add_memory_invalid_category(self, mock_settings, mock_emb, mock_milvus):
        """测试无效 category 默认为 fact"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client.search.return_value = [[]]
        mock_milvus.return_value = mock_client

        mock_emb_svc = MagicMock()
        mock_emb_svc.embed_query.return_value = [0.1] * 1024
        mock_emb.return_value = mock_emb_svc

        manager = MemoryManager()
        result = manager.add_memory("test", "invalid_cat", "session")

        assert result is True
        call_data = mock_client.insert.call_args[1]["data"][0]
        assert call_data["category"] == "fact"

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    def test_search_memories(self, mock_settings, mock_emb, mock_milvus):
        """测试检索相关记忆"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client.search.return_value = [[
            {"distance": 0.9, "entity": {"content": "用户喜欢 Python", "category": "preference", "created_at": 123.0}},
            {"distance": 0.8, "entity": {"content": "使用 Milvus", "category": "fact", "created_at": 124.0}},
        ]]
        mock_milvus.return_value = mock_client

        mock_emb_svc = MagicMock()
        mock_emb_svc.embed_query.return_value = [0.1] * 1024
        mock_emb.return_value = mock_emb_svc

        manager = MemoryManager()
        results = manager.search_memories("Python 偏好", top_k=3)

        assert len(results) == 2
        assert results[0]["score"] == 0.9

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    def test_search_memories_empty(self, mock_settings, mock_emb, mock_milvus):
        """测试记忆库为空时返回空列表"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client.search.return_value = [[]]
        mock_milvus.return_value = mock_client

        mock_emb_svc = MagicMock()
        mock_emb_svc.embed_query.return_value = [0.1] * 1024
        mock_emb.return_value = mock_emb_svc

        manager = MemoryManager()
        results = manager.search_memories("test")

        assert results == []

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    def test_delete_memory(self, mock_settings, mock_emb, mock_milvus):
        """测试删除记忆"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_milvus.return_value = mock_client

        manager = MemoryManager()
        result = manager.delete_memory("test_id")

        assert result is True
        mock_client.delete.assert_called_once()

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    def test_list_memories(self, mock_settings, mock_emb, mock_milvus):
        """测试列出记忆"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client.query.return_value = [
            {"id": "1", "content": "test", "category": "fact", "created_at": 123.0},
        ]
        mock_milvus.return_value = mock_client

        manager = MemoryManager()
        results = manager.list_memories()

        assert len(results) == 1

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    def test_list_memories_with_category(self, mock_settings, mock_emb, mock_milvus):
        """测试按 category 筛选记忆"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client.query.return_value = []
        mock_milvus.return_value = mock_client

        manager = MemoryManager()
        manager.list_memories(category="preference")

        call_kwargs = mock_client.query.call_args[1]
        assert 'category == "preference"' in call_kwargs["filter"]

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    def test_format_memories(self, mock_settings, mock_emb, mock_milvus):
        """测试格式化记忆"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_milvus.return_value = mock_client

        manager = MemoryManager()

        memories = [
            {"content": "用户喜欢 Python", "category": "preference", "score": 0.9},
            {"content": "使用 Milvus", "category": "fact", "score": 0.8},
        ]
        result = manager.format_memories(memories)
        assert "[preference] 用户喜欢 Python" in result
        assert "[fact] 使用 Milvus" in result

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    def test_format_memories_empty(self, mock_settings, mock_emb, mock_milvus):
        """测试空记忆格式化"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_milvus.return_value = mock_client

        manager = MemoryManager()
        assert manager.format_memories([]) == ""
