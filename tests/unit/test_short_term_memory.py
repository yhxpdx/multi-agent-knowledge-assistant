"""
短时记忆修复单元测试

测试覆盖：
1. 对话历史注入 graph
2. 历史截断（最多 20 条）
3. web_search_agent 读取历史
4. agent 读取 context
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from backend.agents.supervisor import supervisor_node, router, build_graph, AgentState
from backend.agents.web_search_agent import web_search_agent
from backend.agents.general_agent import general_agent


class TestHistoryInjection:
    """对话历史注入测试"""

    def test_history_conversion_to_messages(self):
        """测试 Redis 历史转换为 BaseMessage 列表"""
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你的？"},
            {"role": "user", "content": "什么是 RAG？"},
        ]

        messages = []
        for msg in history[-20:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content="继续"))

        assert len(messages) == 4
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "你好"
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == "你好！有什么可以帮你的？"
        assert messages[-1].content == "继续"

    def test_history_truncation(self):
        """测试历史截断到 20 条"""
        history = [{"role": "user", "content": f"msg{i}"} for i in range(30)]

        messages = []
        for msg in history[-20:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
        messages.append(HumanMessage(content="current"))

        assert len(messages) == 21  # 20 history + 1 current
        assert messages[0].content == "msg10"  # starts from msg10 (30-20=10)

    def test_empty_history(self):
        """测试空历史"""
        history = []

        messages = []
        for msg in history[-20:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
        messages.append(HumanMessage(content="hello"))

        assert len(messages) == 1
        assert messages[0].content == "hello"


class TestWebSearchAgentHistory:
    """web_search_agent 对话历史测试"""

    @patch("backend.agents.web_search_agent.get_llm_client")
    @patch("backend.agents.web_search_agent.web_search")
    def test_web_search_reads_history(self, mock_search, mock_get_llm):
        """测试 web_search_agent 读取对话历史"""
        mock_search.invoke.return_value = "搜索结果"
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "回答"
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [
                HumanMessage(content="你好"),
                AIMessage(content="你好！"),
                HumanMessage(content="今天天气怎么样"),
            ],
            "context": "",
        }

        web_search_agent(state)

        call_args = mock_llm.chat.call_args[0][0]
        # 应该有 system + 3 条历史消息
        assert len(call_args) == 4
        assert call_args[1]["role"] == "user"
        assert call_args[1]["content"] == "你好"


class TestAgentContextReading:
    """Agent context 读取测试"""

    @patch("backend.agents.general_agent.get_llm_client")
    def test_general_agent_reads_context(self, mock_get_llm):
        """测试 general_agent 读取 context"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "回答"
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="你好")],
            "context": "以下是与你相关的记忆：\n[preference] 用户喜欢 Python",
        }

        general_agent(state)

        call_args = mock_llm.chat.call_args[0][0]
        system_msg = call_args[0]["content"]
        assert "记忆" in system_msg
        assert "用户喜欢 Python" in system_msg

    @patch("backend.agents.general_agent.get_llm_client")
    def test_general_agent_empty_context(self, mock_get_llm):
        """测试 general_agent 空 context"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "回答"
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="你好")],
            "context": "",
        }

        general_agent(state)

        call_args = mock_llm.chat.call_args[0][0]
        system_msg = call_args[0]["content"]
        assert "记忆" not in system_msg


class TestSupervisorMemorySearch:
    """Supervisor 记忆检索测试"""

    @patch("backend.agents.supervisor.get_llm_client")
    @patch("backend.agents.supervisor.get_memory_manager")
    def test_supervisor_searches_memory(self, mock_mem_mgr, mock_get_llm):
        """测试 Supervisor 检索长期记忆并写入 context"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = '{"agent": "general"}'
        mock_get_llm.return_value = mock_llm

        mock_mgr = MagicMock()
        mock_mgr.search_memories.return_value = [
            {"content": "用户喜欢 Python", "category": "preference", "score": 0.9}
        ]
        mock_mgr.format_memories.return_value = "[preference] 用户喜欢 Python"
        mock_mem_mgr.return_value = mock_mgr

        state: AgentState = {
            "messages": [HumanMessage(content="写个排序算法")],
            "next_agent": "",
            "context": "",
        }

        result = supervisor_node(state)

        assert result["next_agent"] == "general"
        assert "Python" in result["context"]

    @patch("backend.agents.supervisor.get_llm_client")
    @patch("backend.agents.supervisor.get_memory_manager")
    def test_supervisor_no_memory(self, mock_mem_mgr, mock_get_llm):
        """测试无相关记忆时 context 为空"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = '{"agent": "general"}'
        mock_get_llm.return_value = mock_llm

        mock_mgr = MagicMock()
        mock_mgr.search_memories.return_value = []
        mock_mgr.format_memories.return_value = ""
        mock_mem_mgr.return_value = mock_mgr

        state: AgentState = {
            "messages": [HumanMessage(content="你好")],
            "next_agent": "",
            "context": "",
        }

        result = supervisor_node(state)

        assert result["context"] == ""
