"""
Embedding 服务单元测试

测试覆盖：
1. embed_query 正常调用
2. embed_query 异常处理
3. embed_documents 批量处理
4. health_check 服务可用
5. health_check 服务不可用
6. 单例模式
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.core.embedding import EmbeddingService, get_embedding_service


class TestEmbeddingService:
    """Embedding 服务测试"""

    @pytest.fixture
    def service(self):
        """创建服务实例"""
        with patch("backend.core.embedding.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                EMBEDDING_MODEL="bge-m3",
                OLLAMA_HOST="http://localhost:11434",
                EMBEDDING_DIM=1024,
            )
            return EmbeddingService()

    @patch("backend.core.embedding.requests.post")
    def test_embed_query_success(self, mock_post, service):
        """测试正常 embedding 查询"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1] * 1024
        }
        mock_post.return_value = mock_response

        result = service.embed_query("测试文本")
        assert result is not None
        assert len(result) == 1024
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/embeddings",
            json={"model": "bge-m3", "prompt": "测试文本"},
            timeout=30,
        )

    @patch("backend.core.embedding.requests.post")
    def test_embed_query_failure(self, mock_post, service):
        """测试 embedding 查询失败"""
        mock_post.side_effect = Exception("Connection refused")

        result = service.embed_query("测试文本")
        assert result is None

    @patch("backend.core.embedding.requests.post")
    def test_embed_query_http_error(self, mock_post, service):
        """测试 HTTP 错误响应"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = service.embed_query("测试文本")
        assert result is None

    @patch("backend.core.embedding.requests.post")
    def test_embed_documents(self, mock_post, service):
        """测试批量 embedding"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1] * 1024}
        mock_post.return_value = mock_response

        texts = ["文本1", "文本2", "文本3"]
        results = service.embed_documents(texts)

        assert len(results) == 3
        assert all(r is not None for r in results)
        assert mock_post.call_count == 3

    @patch("backend.core.embedding.requests.get")
    def test_health_check_success(self, mock_get, service):
        """测试健康检查 - 服务可用"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert service.health_check() is True

    @patch("backend.core.embedding.requests.get")
    def test_health_check_failure(self, mock_get, service):
        """测试健康检查 - 服务不可用"""
        mock_get.side_effect = Exception("Connection refused")

        assert service.health_check() is False

    def test_singleton(self):
        """测试单例模式"""
        import backend.core.embedding as mod
        mod._embedding_service = None

        with patch("backend.core.embedding.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                EMBEDDING_MODEL="bge-m3",
                OLLAMA_HOST="http://localhost:11434",
                EMBEDDING_DIM=1024,
            )
            s1 = get_embedding_service()
            s2 = get_embedding_service()
            assert s1 is s2
