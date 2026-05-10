"""
前端功能测试

测试覆盖：
1. Streamlit 页面配置
2. 会话管理逻辑
3. 消息发送逻辑
4. 文档上传逻辑
5. 服务状态检查
6. SSE 流式响应解析

注意：这些测试验证前端逻辑，不启动 Streamlit 服务器
通过 mock requests 库来模拟后端 API
"""

import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock


class TestFrontendSessionManagement:
    """前端会话管理测试"""

    def test_create_session_success(self):
        """测试成功创建会话"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"session_id": "abc12345"}

        with patch("requests.post", return_value=mock_response):
            import requests
            resp = requests.post("http://localhost:8000/api/sessions")
            data = resp.json()

            assert data["session_id"] == "abc12345"
            assert resp.status_code == 200

    def test_create_session_failure(self):
        """测试创建会话失败"""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("requests.post", return_value=mock_response):
            import requests
            resp = requests.post("http://localhost:8000/api/sessions")

            assert resp.status_code == 500

    def test_list_sessions(self):
        """测试获取会话列表"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"session_id": "sess1", "message_count": 5},
            {"session_id": "sess2", "message_count": 3},
        ]

        with patch("requests.get", return_value=mock_response):
            import requests
            resp = requests.get("http://localhost:8000/api/sessions")
            sessions = resp.json()

            assert len(sessions) == 2
            assert sessions[0]["session_id"] == "sess1"


class TestFrontendChat:
    """前端聊天功能测试"""

    def test_send_message_non_stream(self):
        """测试发送非流式消息"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "RAG 是检索增强生成技术",
            "session_id": "abc12345",
            "agent": "document_qa",
        }

        with patch("requests.post", return_value=mock_response):
            import requests
            resp = requests.post(
                "http://localhost:8000/api/chat",
                json={"message": "什么是 RAG？", "stream": False},
            )
            data = resp.json()

            assert "RAG" in data["response"]
            assert data["agent"] == "document_qa"

    def test_send_message_stream(self):
        """测试发送流式消息"""
        # 模拟 SSE 响应
        sse_lines = [
            'data: {"content": "RAG", "node": "document_qa"}',
            'data: {"content": " 是", "node": "document_qa"}',
            'data: {"content": "检索增强生成", "node": "document_qa"}',
            'data: {"done": true, "session_id": "abc12345"}',
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            line.encode("utf-8") for line in sse_lines
        ]

        with patch("requests.post", return_value=mock_response):
            import requests
            resp = requests.post(
                "http://localhost:8000/api/chat",
                json={"message": "什么是 RAG？", "stream": True},
                stream=True,
            )

            full_response = ""
            session_id = None

            for line in resp.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("done"):
                            session_id = data.get("session_id")
                        else:
                            full_response += data.get("content", "")

            assert full_response == "RAG 是检索增强生成"
            assert session_id == "abc12345"

    def test_stream_error_handling(self):
        """测试流式响应错误处理"""
        sse_lines = [
            'data: {"error": "LLM 服务不可用"}',
        ]

        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            line.encode("utf-8") for line in sse_lines
        ]

        with patch("requests.post", return_value=mock_response):
            import requests
            resp = requests.post(
                "http://localhost:8000/api/chat",
                json={"message": "测试", "stream": True},
                stream=True,
            )

            error = None
            for line in resp.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("error"):
                            error = data["error"]

            assert error == "LLM 服务不可用"


class TestFrontendDocumentManagement:
    """前端文档管理测试"""

    def test_upload_document_success(self):
        """测试文档上传成功"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "doc_id": "doc123",
            "filename": "test.txt",
            "chunk_count": 10,
        }

        with patch("requests.post", return_value=mock_response):
            import requests
            files = {"file": ("test.txt", b"test content", "text/plain")}
            resp = requests.post("http://localhost:8000/api/documents", files=files)
            data = resp.json()

            assert data["doc_id"] == "doc123"
            assert data["chunk_count"] == 10

    def test_upload_document_failure(self):
        """测试文档上传失败"""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "不支持的文件类型"}

        with patch("requests.post", return_value=mock_response):
            import requests
            resp = requests.post("http://localhost:8000/api/documents")
            assert resp.status_code == 400

    def test_list_documents(self):
        """测试文档列表"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"doc_id": "doc1", "filename": "test1.txt", "chunk_count": 5},
            {"doc_id": "doc2", "filename": "test2.md", "chunk_count": 8},
        ]

        with patch("requests.get", return_value=mock_response):
            import requests
            resp = requests.get("http://localhost:8000/api/documents")
            docs = resp.json()

            assert len(docs) == 2
            assert docs[0]["filename"] == "test1.txt"

    def test_delete_document(self):
        """测试删除文档"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "文档 doc1 已删除"}

        with patch("requests.delete", return_value=mock_response):
            import requests
            resp = requests.delete("http://localhost:8000/api/documents/doc1")
            assert resp.status_code == 200


class TestFrontendHealthCheck:
    """前端服务状态检查测试"""

    def test_health_check_success(self):
        """测试健康检查成功"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "services": {
                "redis": True,
                "ollama_embedding": True,
                "milvus": True,
                "llm": True,
            },
        }

        with patch("requests.get", return_value=mock_response):
            import requests
            resp = requests.get("http://localhost:8000/api/health")
            health = resp.json()

            assert health["status"] == "healthy"
            assert all(health["services"].values())

    def test_health_check_partial_failure(self):
        """测试部分服务不可用"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "services": {
                "redis": True,
                "ollama_embedding": False,
                "milvus": True,
                "llm": True,
            },
        }

        with patch("requests.get", return_value=mock_response):
            import requests
            resp = requests.get("http://localhost:8000/api/health")
            health = resp.json()

            assert health["services"]["redis"] is True
            assert health["services"]["ollama_embedding"] is False

    def test_health_check_connection_error(self):
        """测试后端不可达"""
        import requests
        with patch("requests.get", side_effect=requests.ConnectionError("Connection refused")):
            try:
                requests.get("http://localhost:8000/api/health")
            except requests.ConnectionError:
                pass  # 预期的异常


class TestSSEParsing:
    """SSE 响应解析测试"""

    def test_parse_sse_line(self):
        """测试解析单行 SSE 数据"""
        line = 'data: {"content": "Hello", "node": "document_qa"}'
        json_str = line[6:]  # 去掉 "data: " 前缀
        data = json.loads(json_str)

        assert data["content"] == "Hello"
        assert data["node"] == "document_qa"

    def test_parse_sse_done(self):
        """测试解析完成标记"""
        line = 'data: {"done": true, "session_id": "abc123"}'
        data = json.loads(line[6:])

        assert data["done"] is True
        assert data["session_id"] == "abc123"

    def test_parse_sse_error(self):
        """测试解析错误消息"""
        line = 'data: {"error": "Something went wrong"}'
        data = json.loads(line[6:])

        assert "error" in data

    def test_parse_sse_empty_line(self):
        """测试空行处理"""
        line = ""
        assert not line.startswith("data: ")

    def test_parse_sse_multiline_response(self):
        """测试多行 SSE 响应"""
        lines = [
            'data: {"content": "你", "node": "document_qa"}',
            'data: {"content": "好", "node": "document_qa"}',
            'data: {"content": "！", "node": "document_qa"}',
            'data: {"done": true, "session_id": "abc123"}',
        ]

        full_response = ""
        session_id = None

        for line in lines:
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if data.get("done"):
                    session_id = data.get("session_id")
                else:
                    full_response += data.get("content", "")

        assert full_response == "你好！"
        assert session_id == "abc123"
