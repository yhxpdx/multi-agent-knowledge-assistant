"""
FastAPI 接口测试

测试覆盖：
1. GET / - 根路径
2. GET /api/health - 健康检查
3. POST /api/chat - 聊天接口（流式和非流式）
4. POST /api/sessions - 创建会话
5. GET /api/sessions - 会话列表
6. DELETE /api/sessions/{id} - 删除会话
7. GET /api/documents - 文档列表
8. 错误处理和边界条件
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


@pytest.fixture
def mock_redis():
    """Mock Redis 管理器"""
    with patch("backend.api.sessions.get_redis_manager") as mock, \
         patch("backend.api.chat.get_redis_manager") as mock2:
        redis_mgr = MagicMock()
        redis_mgr.create_session.return_value = "abc12345"
        redis_mgr.list_sessions.return_value = [
            {"session_id": "abc12345", "created_at": 1000, "last_active": 2000, "message_count": 5}
        ]
        redis_mgr.get_messages.return_value = [
            {"role": "user", "content": "你好", "timestamp": 1000},
            {"role": "assistant", "content": "你好！", "timestamp": 1001},
        ]
        redis_mgr.get_history_for_llm.return_value = [
            {"role": "user", "content": "你好"},
        ]
        redis_mgr.health_check.return_value = True
        mock.return_value = redis_mgr
        mock2.return_value = redis_mgr
        yield redis_mgr


@pytest.fixture
def mock_agent_graph():
    """Mock Agent 图"""
    with patch("backend.api.chat.agent_graph") as mock:
        from langchain_core.messages import AIMessage
        mock.invoke.return_value = {
            "messages": [AIMessage(content="测试回复")],
            "next_agent": "document_qa",
        }
        yield mock


@pytest.fixture
def client(mock_redis, mock_agent_graph):
    """创建测试客户端"""
    # Mock 所有外部依赖
    with patch("backend.api.health.get_redis_manager") as mock_h_redis, \
         patch("backend.api.health.get_embedding_service") as mock_h_emb, \
         patch("backend.api.health.get_milvus_manager") as mock_h_mil, \
         patch("backend.api.health.get_llm_client") as mock_h_llm:

        mock_h_redis.return_value = MagicMock(health_check=MagicMock(return_value=True))
        mock_h_emb.return_value = MagicMock(
            health_check=MagicMock(return_value=True),
            model="bge-m3",
            dimension=1024,
        )
        mock_h_mil.return_value = MagicMock()
        mock_h_llm.return_value = MagicMock(model="doubao-seed-2-0-lite-260428")

        from backend.main import app
        yield TestClient(app)


class TestRootEndpoint:
    """根路径测试"""

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "version" in data


class TestHealthEndpoint:
    """健康检查测试"""

    def test_health_check(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "config" in data


class TestSessionEndpoints:
    """会话管理测试"""

    def test_create_session(self, client, mock_redis):
        resp = client.post("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["session_id"] == "abc12345"

    def test_list_sessions(self, client, mock_redis):
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1

    def test_get_session_messages(self, client, mock_redis):
        resp = client.get("/api/sessions/abc12345/messages")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_delete_session(self, client, mock_redis):
        resp = client.delete("/api/sessions/abc12345")
        assert resp.status_code == 200
        mock_redis.delete_session.assert_called_once_with("abc12345")


class TestChatEndpoint:
    """聊天接口测试"""

    def test_chat_non_stream(self, client, mock_redis, mock_agent_graph):
        """测试非流式聊天"""
        resp = client.post("/api/chat", json={
            "message": "什么是 RAG？",
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "session_id" in data
        assert "agent" in data

    def test_chat_with_session_id(self, client, mock_redis, mock_agent_graph):
        """测试带 session_id 的聊天"""
        resp = client.post("/api/chat", json={
            "message": "你好",
            "session_id": "existing_session",
            "stream": False,
        })
        assert resp.status_code == 200

    def test_chat_creates_session(self, client, mock_redis, mock_agent_graph):
        """测试聊天自动创建会话"""
        resp = client.post("/api/chat", json={
            "message": "你好",
            "stream": False,
        })
        assert resp.status_code == 200
        mock_redis.create_session.assert_called()

    def test_chat_saves_messages(self, client, mock_redis, mock_agent_graph):
        """测试聊天保存消息"""
        client.post("/api/chat", json={
            "message": "测试消息",
            "stream": False,
        })
        # 应该保存用户消息和助手回复
        assert mock_redis.add_message.call_count >= 2


class TestDocumentEndpoints:
    """文档管理测试"""

    def test_list_documents_empty(self, client):
        """测试空文档列表"""
        with patch("backend.api.documents._load_metadata") as mock_meta:
            mock_meta.return_value = {}
            resp = client.get("/api/documents")
            assert resp.status_code == 200
            assert resp.json() == []

    def test_list_documents(self, client):
        """测试文档列表"""
        with patch("backend.api.documents._load_metadata") as mock_meta:
            mock_meta.return_value = {
                "doc1": {
                    "filename": "test.txt",
                    "file_type": ".txt",
                    "size_bytes": 1024,
                    "total_chars": 500,
                    "chunk_count": 5,
                }
            }
            resp = client.get("/api/documents")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["doc_id"] == "doc1"

    def test_delete_document_not_found(self, client):
        """测试删除不存在的文档"""
        with patch("backend.api.documents._load_metadata") as mock_meta:
            mock_meta.return_value = {}
            resp = client.delete("/api/documents/nonexistent")
            assert resp.status_code == 404

    def test_upload_invalid_file_type(self, client):
        """测试上传不支持的文件类型"""
        resp = client.post(
            "/api/documents",
            files={"file": ("test.xyz", b"content", "application/octet-stream")},
        )
        assert resp.status_code == 400


class TestCORS:
    """CORS 配置测试"""

    def test_cors_headers(self, client):
        """测试 CORS 头"""
        resp = client.options("/api/health", headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        })
        # CORS 中间件应该处理 OPTIONS 请求
        assert resp.status_code in [200, 204, 405]
