"""
会话管理 API
"""

from fastapi import APIRouter, HTTPException
from backend.core.redis_manager import get_redis_manager

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("")
async def create_session():
    """创建新会话"""
    redis = get_redis_manager()
    session_id = redis.create_session()
    return {"session_id": session_id}


@router.get("")
async def list_sessions():
    """列出所有会话"""
    redis = get_redis_manager()
    return redis.list_sessions()


@router.get("/{session_id}/messages")
async def get_session_messages(session_id: str):
    """获取会话消息历史"""
    redis = get_redis_manager()
    messages = redis.get_messages(session_id)
    if not messages:
        raise HTTPException(404, "会话不存在")
    return messages


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    redis = get_redis_manager()
    redis.delete_session(session_id)
    return {"message": f"会话 {session_id} 已删除"}
