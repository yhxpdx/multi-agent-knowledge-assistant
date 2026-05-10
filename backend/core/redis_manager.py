"""
Redis 对话记忆管理

设计决策（全链路思考）：
1. 为什么用 Redis 存储对话历史？
   - 读写性能好（微秒级响应）
   - 支持 TTL 自动过期
   - 数据结构丰富（List 适合消息序列）
   - 项目已有 Docker 部署

2. 为什么用 List 而不是 String？
   - List 支持 LPUSH/RPOP，天然适合消息队列
   - 可以用 LRANGE 获取最近 N 条消息
   - 不需要每次都序列化/反序列化整个历史

3. 消息格式设计：
   - 每条消息是一个 JSON 字符串
   - 包含 role、content、timestamp
   - timestamp 用于调试和排序
"""

import json
import time
import uuid
import redis
from typing import Optional
from backend.core.config import get_settings


class RedisManager:
    def __init__(self):
        self.settings = get_settings()
        self.client = redis.Redis(
            host=self.settings.REDIS_HOST,
            port=self.settings.REDIS_PORT,
            db=self.settings.REDIS_DB,
            decode_responses=True,
        )
        self.ttl = self.settings.SESSION_TTL_SECONDS
        self.max_messages = self.settings.MAX_HISTORY_TURNS * 2  # 每轮 2 条消息

    def create_session(self) -> str:
        """创建新会话，返回 session_id"""
        session_id = str(uuid.uuid4())[:8]
        key = f"chat:{session_id}:meta"
        self.client.hset(key, mapping={
            "created_at": str(int(time.time())),
            "last_active": str(int(time.time())),
        })
        self.client.expire(key, self.ttl)
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        """添加一条消息到会话"""
        key = f"chat:{session_id}:messages"
        message = json.dumps({
            "role": role,
            "content": content,
            "timestamp": int(time.time()),
        }, ensure_ascii=False)
        self.client.rpush(key, message)
        self.client.expire(key, self.ttl)

        # 更新最后活跃时间
        meta_key = f"chat:{session_id}:meta"
        self.client.hset(meta_key, "last_active", str(int(time.time())))

        # 截断超过限制的消息
        length = self.client.llen(key)
        if length > self.max_messages:
            self.client.ltrim(key, length - self.max_messages, -1)

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> list[dict]:
        """获取会话的消息历史"""
        key = f"chat:{session_id}:messages"
        if limit:
            # 获取最近 limit 条
            length = self.client.llen(key)
            start = max(0, length - limit)
            raw = self.client.lrange(key, start, -1)
        else:
            raw = self.client.lrange(key, 0, -1)

        return [json.loads(msg) for msg in raw]

    def get_history_for_llm(self, session_id: str) -> list[dict]:
        """获取格式化的对话历史（用于 LLM 输入）"""
        messages = self.get_messages(session_id, limit=self.max_messages)
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    def list_sessions(self) -> list[dict]:
        """列出所有会话"""
        pattern = "chat:*:meta"
        keys = self.client.keys(pattern)
        sessions = []
        for key in keys:
            session_id = key.split(":")[1]
            meta = self.client.hgetall(key)
            msg_key = f"chat:{session_id}:messages"
            msg_count = self.client.llen(msg_key)
            sessions.append({
                "session_id": session_id,
                "created_at": int(meta.get("created_at", 0)),
                "last_active": int(meta.get("last_active", 0)),
                "message_count": msg_count,
            })
        sessions.sort(key=lambda x: x["last_active"], reverse=True)
        return sessions

    def delete_session(self, session_id: str):
        """删除会话"""
        self.client.delete(f"chat:{session_id}:messages")
        self.client.delete(f"chat:{session_id}:meta")

    def health_check(self) -> bool:
        """检查 Redis 连接"""
        try:
            return self.client.ping()
        except:
            return False


# 全局单例
_redis_manager: Optional[RedisManager] = None


def get_redis_manager() -> RedisManager:
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisManager()
    return _redis_manager
