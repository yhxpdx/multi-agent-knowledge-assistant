"""
Code Assistant Agent — 代码助手智能体

工作流程：
1. 分析用户的代码需求（写代码、解释代码、调试等）
2. 生成代码并可选择执行验证
3. 解释代码逻辑
"""

from langchain_core.messages import HumanMessage, AIMessage

from backend.core.llm_client import get_llm_client
from backend.tools.code_executor import code_executor
from backend.tools.calculator import calculator


CODE_ASSISTANT_PROMPT = """你是一个专业的编程助手。你可以：
1. 编写代码解决问题
2. 解释代码逻辑
3. 调试和修复代码
4. 进行数学计算

当用户要求执行代码时，使用 code_executor 工具执行。
当用户要求计算时，使用 calculator 工具。
"""


def code_assistant_agent(state: dict) -> dict:
    """Code Assistant Agent 节点"""
    llm = get_llm_client()
    messages = state["messages"]

    # 获取用户最后一条消息
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    # 构建消息
    llm_messages = [
        {"role": "system", "content": CODE_ASSISTANT_PROMPT},
    ]

    # 添加对话历史
    for msg in messages[-6:]:
        if isinstance(msg, HumanMessage):
            llm_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            llm_messages.append({"role": "assistant", "content": msg.content})

    # 先让 LLM 生成回答
    response = llm.chat(llm_messages, temperature=0.7)

    # 检查是否需要执行代码
    if any(kw in user_query.lower() for kw in ["执行", "运行", "计算", "算一下", "execute", "run", "calculate"]):
        # 尝试从响应中提取代码
        code = _extract_code(response)
        if code:
            exec_result = code_executor.invoke({"code": code})
            response += f"\n\n**执行结果：**\n```\n{exec_result}\n```"

    return {"messages": [AIMessage(content=response)]}


def _extract_code(text: str) -> str:
    """从 LLM 回答中提取代码块"""
    import re
    # 匹配 ```python ... ``` 或 ``` ... ```
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0].strip() if matches else ""
