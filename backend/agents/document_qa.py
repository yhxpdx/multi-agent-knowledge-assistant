"""
Document QA Agent — 基于 RAG 的文档问答

工作流程：
1. 从用户消息中提取查询
2. 调用 document_search 工具检索相关文档
3. 将检索结果作为上下文，结合用户问题生成回答
4. 引用来源，减少幻觉
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from backend.core.llm_client import get_llm_client
from backend.tools.document_search import document_search


DOCUMENT_QA_PROMPT = """你是一个专业的文档问答助手。你的任务是基于检索到的文档片段回答用户的问题。

回答要求：
1. 只基于提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，明确告知用户
3. 引用来源，在回答中标注信息来自哪个文档
4. 回答要准确、简洁、有条理
"""


def document_qa_agent(state: dict) -> dict:
    """Document QA Agent 节点"""
    llm = get_llm_client()
    messages = state["messages"]

    # 获取用户最后一条消息
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    # 调用文档搜索工具
    search_result = document_search.invoke({"query": user_query, "top_k": 5})

    # 构建带上下文的 prompt
    context = f"以下是检索到的相关文档片段：\n\n{search_result}"

    # 构建消息列表
    llm_messages = [
        {"role": "system", "content": DOCUMENT_QA_PROMPT + "\n\n" + context},
    ]

    # 添加对话历史（最近几轮）
    for msg in messages[-6:]:  # 最近 3 轮
        if isinstance(msg, HumanMessage):
            llm_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            llm_messages.append({"role": "assistant", "content": msg.content})

    # 生成回答
    response = llm.chat(llm_messages, temperature=0.7)

    return {"messages": [AIMessage(content=response)]}
