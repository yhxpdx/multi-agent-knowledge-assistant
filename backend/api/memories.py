"""
记忆管理 API

提供记忆的查看、筛选和删除功能。
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from backend.core.memory_manager import get_memory_manager

router = APIRouter(prefix="/api/memories", tags=["memories"])


@router.get("")
async def list_memories(category: Optional[str] = Query(None, description="按分类筛选")):
    """列出所有记忆"""
    try:
        memories = get_memory_manager().list_memories(category=category)
        return memories
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str):
    """删除指定记忆"""
    try:
        success = get_memory_manager().delete_memory(memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="记忆不存在")
        return {"status": "deleted", "memory_id": memory_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
