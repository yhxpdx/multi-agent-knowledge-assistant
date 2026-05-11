"""
Supervisor Agent — 多智能体编排核心

架构设计（全链路思考）：
1. 为什么选 Supervisor 模式？
   - 面试高频考点，最常见的多智能体架构
   - 职责清晰：Supervisor 负责路由，子 Agent 负责执行
   - 可扩展性好：新增子 Agent 只需注册
   - 代码量适中，适合简历项目展示

2. Supervisor 如何做路由决策？
   - 使用 LLM 分析用户意图
   - 根据关键词和语义判断应该调用哪个 Agent
   - 输出结构化的路由决策（JSON 格式）

3. 状态管理：
   - 使用 TypedDict 定义状态结构
   - 状态在整个图执行过程中共享
   - 包含消息历史、当前 Agent、工具结果等
"""

from typing import Literal, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from backend.core.llm_client import get_llm_client
from backend.core.memory_manager import get_memory_manager
from backend.agents.document_qa import document_qa_agent
from backend.agents.web_search_agent import web_search_agent
from backend.agents.code_assistant import code_assistant_agent
from backend.agents.general_agent import general_agent


# 状态定义
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    context: str


# Supervisor 的系统提示
SUPERVISOR_PROMPT = """你是一个智能助手的路由器。根据用户的问题，判断应该由哪个专业 Agent 来处理。

可用的 Agent：
- document_qa: 处理与已上传文档、知识库相关的问题（如"总结文档"、"什么是 RAG"等）
- web_search: 需要实时信息或联网搜索的问题（如"今天的新闻"、"最新消息"等）
- code_assistant: 代码相关的问题（如"写代码"、"解释代码"、"计算"等）
- general: 日常闲聊、问候、常识性问题等不属于以上三类的对话（如"你好"、"你是谁"、"讲个笑话"等）

请只返回一个 JSON：{"agent": "agent_name"}
不要返回其他内容。"""


def supervisor_node(state: AgentState) -> dict:
    """Supervisor 节点：分析用户意图，检索长期记忆，决定路由"""
    llm = get_llm_client()

    # 获取最后一条用户消息
    messages = state["messages"]
    user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break

    # 检索相关长期记忆，写入 context
    context = ""
    try:
        memory_manager = get_memory_manager()
        memories = memory_manager.search_memories(user_msg, top_k=3)
        context = memory_manager.format_memories(memories)
    except Exception:
        pass  # 记忆检索失败不影响路由

    # 调用 LLM 判断意图
    response = llm.chat([
        {"role": "system", "content": SUPERVISOR_PROMPT},
        {"role": "user", "content": user_msg},
    ], temperature=0, max_tokens=50)

    # 解析路由决策
    import json
    try:
        # 提取 JSON
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        decision = json.loads(response)
        agent = decision.get("agent", "document_qa")
    except:
        agent = "general"  # 默认路由到通用对话

    # 验证 agent 名称
    valid_agents = ["document_qa", "web_search", "code_assistant", "general"]
    if agent not in valid_agents:
        agent = "general"

    return {"next_agent": agent, "context": context}


def router(state: AgentState) -> str:
    """路由函数：根据 next_agent 字段选择下一个节点"""
    return state.get("next_agent", "general")


def build_graph():
    """构建 LangGraph 状态图"""
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("document_qa", document_qa_agent)
    graph.add_node("web_search", web_search_agent)
    graph.add_node("code_assistant", code_assistant_agent)
    graph.add_node("general", general_agent)

    # 设置入口
    graph.set_entry_point("supervisor")

    # 添加条件边：supervisor → 子 agent
    graph.add_conditional_edges(
        "supervisor",
        router,
        {
            "document_qa": "document_qa",
            "web_search": "web_search",
            "code_assistant": "code_assistant",
            "general": "general",
        },
    )

    # 子 agent 完成后结束
    graph.add_edge("document_qa", END)
    graph.add_edge("web_search", END)
    graph.add_edge("code_assistant", END)
    graph.add_edge("general", END)

    return graph.compile()


# 编译图
agent_graph = build_graph()
