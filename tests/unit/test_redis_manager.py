"""
Redis 管理器单元测试

测试覆盖：
1. 会话创建
2. 消息添加
3. 消息获取
4. LLM 历史格式化
5. 会话列表
6. 会话删除
7. 消息截断
8. TTL 设置
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from backend.core.redis_manager import RedisManager, get_redis_manager


class TestRedisManager:
    """Redis 管理器测试"""

    @pytest.fixture
    def manager(self):
        """创建管理器实例"""
        with patch("backend.core.redis_manager.get_settings") as mock_settings, \
             patch("backend.core.redis_manager.redis.Redis") as mock_redis:
            mock_settings.return_value = MagicMock(
                REDIS_HOST="localhost",
                REDIS_PORT=6379,
                REDIS_DB=0,
                SESSION_TTL_SECONDS=604800,
                MAX_HISTORY_TURNS=10,
            )
            mgr = RedisManager()
            mgr._mock_redis = mock_redis
            return mgr

    def test_create_session(self, manager):
        """测试创建会话"""
        session_id = manager.create_session()

        assert session_id is not None
        assert len(session_id) == 8
        manager.client.hset.assert_called_once()
        manager.client.expire.assert_called_once()

    def test_add_message(self, manager):
        """测试添加消息"""
        manager.client.llen.return_value = 2

        manager.add_message("test_session", "user", "你好")

        manager.client.rpush.assert_called_once()
        manager.client.expire.assert_called_once()
        manager.client.hset.assert_called()  # 更新 last_active

    def test_add_message_truncation(self, manager):
        """测试消息截断"""
        manager.client.llen.return_value = 25  # 超过 max_messages=20

        manager.add_message("test_session", "user", "消息")

        manager.client.ltrim.assert_called_once()

    def test_get_messages_all(self, manager):
        """测试获取所有消息"""
        mock_messages = [
            json.dumps({"role": "user", "content": "你好", "timestamp": 1000}, ensure_ascii=False),
            json.dumps({"role": "assistant", "content": "你好！有什么可以帮助你的？", "timestamp": 1001}, ensure_ascii=False),
        ]
        manager.client.lrange.return_value = mock_messages

        messages = manager.get_messages("test_session")

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_get_messages_with_limit(self, manager):
        """测试获取有限条消息"""
        manager.client.llen.return_value = 10
        manager.client.lrange.return_value = [
            json.dumps({"role": "user", "content": "消息5", "timestamp": 1005}, ensure_ascii=False),
        ]

        messages = manager.get_messages("test_session", limit=1)

        assert len(messages) == 1
        manager.client.lrange.assert_called_with("chat:test_session:messages", 9, -1)

    def test_get_history_for_llm(self, manager):
        """测试获取 LLM 格式的历史"""
        mock_messages = [
            json.dumps({"role": "user", "content": "你好", "timestamp": 1000}, ensure_ascii=False),
            json.dumps({"role": "assistant", "content": "你好！", "timestamp": 1001}, ensure_ascii=False),
        ]
        manager.client.llen.return_value = 2
        manager.client.lrange.return_value = mock_messages

        history = manager.get_history_for_llm("test_session")

        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "你好"}
        assert history[1] == {"role": "assistant", "content": "你好！"}

    def test_list_sessions(self, manager):
        """测试列出会话"""
        manager.client.keys.return_value = ["chat:sess1:meta", "chat:sess2:meta"]
        manager.client.hgetall.side_effect = [
            {"created_at": "1000", "last_active": "2000"},
            {"created_at": "1500", "last_active": "3000"},
        ]
        manager.client.llen.side_effect = [5, 3]

        sessions = manager.list_sessions()

        assert len(sessions) == 2
        # 应该按 last_active 降序排列
        assert sessions[0]["last_active"] >= sessions[1]["last_active"]

    def test_delete_session(self, manager):
        """测试删除会话"""
        manager.delete_session("test_session")

        assert manager.client.delete.call_count == 2

    def test_health_check_success(self, manager):
        """测试健康检查 - 连接正常"""
        manager.client.ping.return_value = True

        assert manager.health_check() is True

    def test_health_check_failure(self, manager):
        """测试健康检查 - 连接失败"""
        manager.client.ping.side_effect = Exception("Connection refused")

        assert manager.health_check() is False

    def test_singleton(self):
        """测试单例模式"""
        import backend.core.redis_manager as mod
        mod._redis_manager = None

        with patch("backend.core.redis_manager.get_settings") as mock_settings, \
             patch("backend.core.redis_manager.redis.Redis"):
            mock_settings.return_value = MagicMock(
                REDIS_HOST="localhost",
                REDIS_PORT=6379,
                REDIS_DB=0,
                SESSION_TTL_SECONDS=604800,
                MAX_HISTORY_TURNS=10,
            )
            m1 = get_redis_manager()
            m2 = get_redis_manager()
            assert m1 is m2
