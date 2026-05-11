"""
记忆提取单元测试

测试覆盖：
1. 提取成功
2. should_save=false 不提取
3. JSON 解析失败容错
4. LLM 调用失败容错
5. category 校验与默认值
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.core.memory_manager import MemoryManager


class TestMemoryExtraction:
    """记忆提取测试"""

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    @patch("backend.core.memory_manager.get_llm_client")
    def test_extract_success(self, mock_llm, mock_settings, mock_emb, mock_milvus):
        """测试成功提取记忆"""
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

        mock_llm_instance = MagicMock()
        mock_llm_instance.chat.return_value = '{"should_save": true, "memory": "用户偏好 Python", "category": "preference"}'
        mock_llm.return_value = mock_llm_instance

        manager = MemoryManager()
        result = manager.extract_memory("我喜欢用 Python", "好的，我会记住你喜欢 Python", "test_session")

        assert result is True
        mock_client.insert.assert_called_once()

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    @patch("backend.core.memory_manager.get_llm_client")
    def test_extract_no_save(self, mock_llm, mock_settings, mock_emb, mock_milvus):
        """测试 should_save=false 不提取"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_milvus.return_value = mock_client

        mock_emb_svc = MagicMock()
        mock_emb.return_value = mock_emb_svc

        mock_llm_instance = MagicMock()
        mock_llm_instance.chat.return_value = '{"should_save": false, "memory": "", "category": ""}'
        mock_llm.return_value = mock_llm_instance

        manager = MemoryManager()
        result = manager.extract_memory("你好", "你好！", "test_session")

        assert result is False
        mock_client.insert.assert_not_called()

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    @patch("backend.core.memory_manager.get_llm_client")
    def test_extract_json_error(self, mock_llm, mock_settings, mock_emb, mock_milvus):
        """测试 JSON 解析失败容错"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_milvus.return_value = mock_client

        mock_emb_svc = MagicMock()
        mock_emb.return_value = mock_emb_svc

        mock_llm_instance = MagicMock()
        mock_llm_instance.chat.return_value = "我无法解析这个"
        mock_llm.return_value = mock_llm_instance

        manager = MemoryManager()
        result = manager.extract_memory("test", "response", "session")

        assert result is False

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    @patch("backend.core.memory_manager.get_llm_client")
    def test_extract_llm_error(self, mock_llm, mock_settings, mock_emb, mock_milvus):
        """测试 LLM 调用失败容错"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_milvus.return_value = mock_client

        mock_emb_svc = MagicMock()
        mock_emb.return_value = mock_emb_svc

        mock_llm_instance = MagicMock()
        mock_llm_instance.chat.side_effect = Exception("LLM 调用失败")
        mock_llm.return_value = mock_llm_instance

        manager = MemoryManager()
        result = manager.extract_memory("test", "response", "session")

        assert result is False

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    @patch("backend.core.memory_manager.get_llm_client")
    def test_extract_invalid_category(self, mock_llm, mock_settings, mock_emb, mock_milvus):
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

        mock_llm_instance = MagicMock()
        mock_llm_instance.chat.return_value = '{"should_save": true, "memory": "test memory", "category": "unknown"}'
        mock_llm.return_value = mock_llm_instance

        manager = MemoryManager()
        result = manager.extract_memory("test", "response", "session")

        assert result is True
        call_data = mock_client.insert.call_args[1]["data"][0]
        assert call_data["category"] == "fact"

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    @patch("backend.core.memory_manager.get_llm_client")
    def test_extract_empty_memory(self, mock_llm, mock_settings, mock_emb, mock_milvus):
        """测试空 memory 内容不提取"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_milvus.return_value = mock_client

        mock_emb_svc = MagicMock()
        mock_emb.return_value = mock_emb_svc

        mock_llm_instance = MagicMock()
        mock_llm_instance.chat.return_value = '{"should_save": true, "memory": "", "category": "fact"}'
        mock_llm.return_value = mock_llm_instance

        manager = MemoryManager()
        result = manager.extract_memory("test", "response", "session")

        assert result is False
