"""
记忆管理 API 测试

测试覆盖：
1. GET /api/memories — 列出记忆
2. GET /api/memories?category=preference — 按分类筛选
3. DELETE /api/memories/{id} — 删除记忆
4. DELETE /api/memories/{id} — 404 处理
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


class TestMemoriesAPI:
    """记忆管理 API 测试"""

    @patch("backend.api.memories.get_memory_manager")
    def test_list_memories_empty(self, mock_get_mgr):
        """测试列出空记忆"""
        mock_mgr = MagicMock()
        mock_mgr.list_memories.return_value = []
        mock_get_mgr.return_value = mock_mgr

        resp = client.get("/api/memories")
        assert resp.status_code == 200
        assert resp.json() == []

    @patch("backend.api.memories.get_memory_manager")
    def test_list_memories(self, mock_get_mgr):
        """测试列出记忆"""
        mock_mgr = MagicMock()
        mock_mgr.list_memories.return_value = [
            {"id": "1", "content": "用户喜欢 Python", "category": "preference", "created_at": 123.0},
            {"id": "2", "content": "使用 Milvus", "category": "fact", "created_at": 124.0},
        ]
        mock_get_mgr.return_value = mock_mgr

        resp = client.get("/api/memories")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    @patch("backend.api.memories.get_memory_manager")
    def test_list_memories_by_category(self, mock_get_mgr):
        """测试按分类筛选记忆"""
        mock_mgr = MagicMock()
        mock_mgr.list_memories.return_value = [
            {"id": "1", "content": "用户喜欢 Python", "category": "preference", "created_at": 123.0},
        ]
        mock_get_mgr.return_value = mock_mgr

        resp = client.get("/api/memories?category=preference")
        assert resp.status_code == 200
        mock_mgr.list_memories.assert_called_with(category="preference")

    @patch("backend.api.memories.get_memory_manager")
    def test_delete_memory_success(self, mock_get_mgr):
        """测试删除记忆"""
        mock_mgr = MagicMock()
        mock_mgr.delete_memory.return_value = True
        mock_get_mgr.return_value = mock_mgr

        resp = client.delete("/api/memories/test_id")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    @patch("backend.api.memories.get_memory_manager")
    def test_delete_memory_not_found(self, mock_get_mgr):
        """测试删除不存在的记忆"""
        mock_mgr = MagicMock()
        mock_mgr.delete_memory.return_value = False
        mock_get_mgr.return_value = mock_mgr

        resp = client.delete("/api/memories/nonexistent")
        assert resp.status_code == 404
