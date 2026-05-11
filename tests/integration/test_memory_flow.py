"""
记忆系统集成测试

测试覆盖：完整流程（对话→提取→存储→检索→注入）
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from backend.core.memory_manager import MemoryManager


@pytest.mark.integration
class TestMemoryFlow:
    """记忆系统完整流程测试"""

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    @patch("backend.core.memory_manager.get_llm_client")
    def test_full_memory_flow(self, mock_llm, mock_settings, mock_emb, mock_milvus):
        """完整流程：提取→存储→检索→格式化"""
        mock_s = MagicMock()
        mock_s.MILVUS_HOST = "localhost"
        mock_s.MILVUS_PORT = 19530
        mock_s.EMBEDDING_DIM = 1024
        mock_settings.return_value = mock_s

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_milvus.return_value = mock_client

        mock_emb_svc = MagicMock()
        mock_emb_svc.embed_query.return_value = [0.1] * 1024
        mock_emb.return_value = mock_emb_svc

        # Step 1: 提取记忆
        mock_llm_instance = MagicMock()
        mock_llm_instance.chat.return_value = '{"should_save": true, "memory": "用户偏好使用 Python 编程", "category": "preference"}'
        mock_llm.return_value = mock_llm_instance

        # 模拟没有重复记忆
        mock_client.search.return_value = [[]]

        manager = MemoryManager()

        # 提取
        extracted = manager.extract_memory("我喜欢用 Python", "好的，记住了", "session1")
        assert extracted is True
        mock_client.insert.assert_called_once()

        # Step 2: 检索
        mock_client.search.return_value = [[{
            "distance": 0.92,
            "entity": {"content": "用户偏好使用 Python 编程", "category": "preference", "created_at": 123.0},
        }]]

        results = manager.search_memories("Python 编程")
        assert len(results) == 1
        assert results[0]["category"] == "preference"

        # Step 3: 格式化
        formatted = manager.format_memories(results)
        assert "preference" in formatted
        assert "Python" in formatted

    @patch("backend.core.memory_manager.MilvusClient")
    @patch("backend.core.memory_manager.get_embedding_service")
    @patch("backend.core.memory_manager.get_settings")
    @patch("backend.core.memory_manager.get_llm_client")
    def test_memory_extraction_failure_no_effect(self, mock_llm, mock_settings, mock_emb, mock_milvus):
        """记忆提取失败不影响主流程"""
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
        mock_llm_instance.chat.side_effect = Exception("LLM 故障")
        mock_llm.return_value = mock_llm_instance

        manager = MemoryManager()

        # 提取失败应该返回 False，不抛异常
        result = manager.extract_memory("test", "response", "session")
        assert result is False
