"""
聊天 API — SSE 流式响应

设计决策（全链路思考）：
1. 为什么用 SSE 而不是 WebSocket？
   - SSE 是单向推送，适合 LLM 流式输出场景
   - 实现简单，基于 HTTP，无需额外协议
   - FastAPI 通过 sse-starlette 原生支持

2. 流式响应格式：
   - 每个 token 作为一个 event
   - 使用 JSON 格式便于前端解析
   - 最后发送 [DONE] 标记
"""

import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from backend.agents.supervisor import agent_graph
from backend.core.redis_manager import get_redis_manager
from backend.core.memory_manager import get_memory_manager

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    stream: bool = True


class ChatResponse(BaseModel):
    response: str
    session_id: str
    agent: str


@router.post("/chat")
async def chat(request: ChatRequest):
    """聊天接口"""
    redis = get_redis_manager()

    # 获取或创建会话
    session_id = request.session_id
    if not session_id:
        session_id = redis.create_session()

    # 保存用户消息
    redis.add_message(session_id, "user", request.message)

    # 获取对话历史并注入 graph（限制最近 20 条）
    history = redis.get_history_for_llm(session_id)
    messages = []
    for msg in history[-20:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=request.message))

    if request.stream:
        # 流式响应
        async def generate():
            full_response = ""
            try:
                for event in agent_graph.stream({"messages": messages}):
                    for node, output in event.items():
                        if "messages" in output:
                            for msg in output["messages"]:
                                content = msg.content if hasattr(msg, "content") else str(msg)
                                full_response += content
                                yield f"data: {json.dumps({'content': content, 'node': node}, ensure_ascii=False)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

            # 保存助手回复并提取记忆
            if full_response:
                redis.add_message(session_id, "assistant", full_response)
                try:
                    get_memory_manager().extract_memory(request.message, full_response, session_id)
                except Exception:
                    pass  # 记忆提取失败不影响主流程

            yield f"data: {json.dumps({'done': True, 'session_id': session_id}, ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        # 非流式响应
        result = agent_graph.invoke({"messages": messages})
        response_content = ""
        agent_used = result.get("next_agent", "unknown")

        # 从 messages 列表中获取最后一条 AI 消息
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.type == "ai":
                response_content = msg.content
                break

        # 保存助手回复并提取记忆
        redis.add_message(session_id, "assistant", response_content)
        try:
            get_memory_manager().extract_memory(request.message, response_content, session_id)
        except Exception:
            pass

        return ChatResponse(
            response=response_content,
            session_id=session_id,
            agent=agent_used,
        )
