"""
LLM 客户端单元测试

测试覆盖：
1. 非流式对话
2. 流式对话
3. 异常处理
4. 健康检查
5. 单例模式
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.core.llm_client import LLMClient, get_llm_client


class TestLLMClient:
    """LLM 客户端测试"""

    @pytest.fixture
    def client(self):
        """创建客户端实例"""
        with patch("backend.core.llm_client.get_settings") as mock_settings, \
             patch("backend.core.llm_client.OpenAI") as mock_openai:
            mock_settings.return_value = MagicMock(
                ARK_API_KEY="test_key",
                ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/coding/v3",
                ARK_MODEL_ID="doubao-seed-2-0-lite-260428",
            )
            llm = LLMClient()
            llm._mock_openai = mock_openai
            return llm

    def test_chat_success(self, client):
        """测试正常对话"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="测试回复"))]
        client.client.chat.completions.create.return_value = mock_response

        result = client.chat([{"role": "user", "content": "你好"}])

        assert result == "测试回复"
        client.client.chat.completions.create.assert_called_once()

    def test_chat_with_temperature(self, client):
        """测试自定义温度参数"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="回复"))]
        client.client.chat.completions.create.return_value = mock_response

        client.chat([{"role": "user", "content": "你好"}], temperature=0.5, max_tokens=100)

        call_kwargs = client.client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    def test_chat_failure(self, client):
        """测试对话失败"""
        client.client.chat.completions.create.side_effect = Exception("API Error")

        result = client.chat([{"role": "user", "content": "你好"}])

        assert "LLM 调用失败" in result

    def test_chat_stream(self, client):
        """测试流式对话"""
        mock_chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="你"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="好"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="！"))]),
        ]
        client.client.chat.completions.create.return_value = iter(mock_chunks)

        tokens = list(client.chat_stream([{"role": "user", "content": "你好"}]))

        assert tokens == ["你", "好", "！"]

    def test_chat_stream_failure(self, client):
        """测试流式对话失败"""
        client.client.chat.completions.create.side_effect = Exception("API Error")

        tokens = list(client.chat_stream([{"role": "user", "content": "你好"}]))

        assert len(tokens) == 1
        assert "LLM 调用失败" in tokens[0]

    @patch.object(LLMClient, "chat")
    def test_health_check_success(self, mock_chat, client):
        """测试健康检查 - 服务可用"""
        mock_chat.return_value = "Hello!"

        assert client.health_check() is True

    @patch.object(LLMClient, "chat")
    def test_health_check_failure(self, mock_chat, client):
        """测试健康检查 - 服务不可用"""
        mock_chat.return_value = "LLM 调用失败: Connection refused"

        assert client.health_check() is False

    def test_singleton(self):
        """测试单例模式"""
        import backend.core.llm_client as mod
        mod._llm_client = None

        with patch("backend.core.llm_client.get_settings") as mock_settings, \
             patch("backend.core.llm_client.OpenAI"):
            mock_settings.return_value = MagicMock(
                ARK_API_KEY="test",
                ARK_BASE_URL="https://test.com",
                ARK_MODEL_ID="test-model",
            )
            c1 = get_llm_client()
            c2 = get_llm_client()
            assert c1 is c2
