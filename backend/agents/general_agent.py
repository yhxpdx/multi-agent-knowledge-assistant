"""
General Agent — 通用对话（闲聊、问候、常识等）

当用户的消息不属于文档问答、联网搜索、代码助手时，
由该 Agent 直接用 LLM 回答，无需检索外部信息。
"""

from langchain_core.messages import HumanMessage, AIMessage

from backend.core.llm_client import get_llm_client


GENERAL_PROMPT = """你是一个友好、专业的 AI 助手。你可以回答日常闲聊、问候、常识性问题等。
回答要简洁友好。如果用户的问题更适合其他专业领域（如文档查询、代码编写），可以建议用户换一种方式提问。"""


def general_agent(state: dict) -> dict:
    """General Agent 节点：处理通用对话"""
    llm = get_llm_client()
    messages = state["messages"]

    system_prompt = GENERAL_PROMPT
    memory_context = state.get("context", "")
    if memory_context:
        system_prompt += f"\n\n以下是从记忆中检索到的相关信息：\n{memory_context}"

    llm_messages = [
        {"role": "system", "content": system_prompt},
    ]

    for msg in messages[-6:]:
        if isinstance(msg, HumanMessage):
            llm_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            llm_messages.append({"role": "assistant", "content": msg.content})

    response = llm.chat(llm_messages, temperature=0.7)

    return {"messages": [AIMessage(content=response)]}
