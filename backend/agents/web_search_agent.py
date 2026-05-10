"""
Web Search Agent — 联网搜索智能体

工作流程：
1. 从用户消息中提取搜索查询
2. 调用 web_search 工具获取搜索结果
3. 基于搜索结果生成总结回答
"""

from langchain_core.messages import HumanMessage, AIMessage

from backend.core.llm_client import get_llm_client
from backend.tools.web_search import web_search


WEB_SEARCH_PROMPT = """你是一个联网搜索助手。你的任务是基于搜索结果回答用户的问题。

回答要求：
1. 基于搜索结果总结回答，标注信息来源
2. 如果搜索结果不足以回答，如实告知
3. 回答要简洁明了，重点突出
"""


def web_search_agent(state: dict) -> dict:
    """Web Search Agent 节点"""
    llm = get_llm_client()
    messages = state["messages"]

    # 获取用户最后一条消息
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    # 调用搜索工具
    search_result = web_search.invoke({"query": user_query})

    # 构建带搜索结果的 prompt
    context = f"以下是搜索结果：\n\n{search_result}"

    llm_messages = [
        {"role": "system", "content": WEB_SEARCH_PROMPT + "\n\n" + context},
        {"role": "user", "content": user_query},
    ]

    response = llm.chat(llm_messages, temperature=0.7)

    return {"messages": [AIMessage(content=response)]}
