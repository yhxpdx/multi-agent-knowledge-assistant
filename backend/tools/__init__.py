"""
工具注册中心

设计决策：
- 统一管理所有可用工具
- Agent 可以根据需要选择工具子集
"""

from backend.tools.document_search import document_search
from backend.tools.web_search import web_search
from backend.tools.code_executor import code_executor
from backend.tools.calculator import calculator

# 所有可用工具
ALL_TOOLS = [document_search, web_search, code_executor, calculator]

# 按 Agent 分配的工具映射
AGENT_TOOLS = {
    "document_qa": [document_search],
    "web_search": [web_search],
    "code_assistant": [code_executor, calculator],
}

# 工具名称到工具对象的映射
TOOL_MAP = {t.name: t for t in ALL_TOOLS}

__all__ = ["ALL_TOOLS", "AGENT_TOOLS", "TOOL_MAP"]
