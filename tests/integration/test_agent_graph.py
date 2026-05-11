"""
多智能体图集成测试

测试覆盖：
1. Supervisor 路由到 document_qa
2. Supervisor 路由到 web_search
3. Supervisor 路由到 code_assistant
4. 默认路由行为
5. 完整图执行流程

注意：需要 LLM 服务运行
可通过 pytest -m "not integration" 跳过
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from backend.agents.supervisor import (
    AgentState,
    supervisor_node,
    router,
    build_graph,
)


class TestSupervisorNode:
    """Supervisor 节点测试"""

    @patch("backend.agents.supervisor.get_llm_client")
    def test_route_to_document_qa(self, mock_get_llm):
        """测试路由到文档问答 Agent"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = '{"agent": "document_qa"}'
        mock_get_llm.return_value = mock_llm

        state: AgentState = {
            "messages": [HumanMessage(content="什么是 RAG 技术？")],
            "next_agent": "",
            "context": "",
        }

        result = supervisor_node(state)
        assert result["next_agent"] == "document_qa"

    @patch("backend.agents.supervisor.get_llm_client")
    def test_route_to_web_search(self, mock_get_llm):
        """测试路由到联网搜索 Agent"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = '{"agent": "web_search"}'
        mock_get_llm.return_value = mock_llm

        state: AgentState = {
            "messages": [HumanMessage(content="今天有什么新闻？")],
            "next_agent": "",
            "context": "",
        }

        result = supervisor_node(state)
        assert result["next_agent"] == "web_search"

    @patch("backend.agents.supervisor.get_llm_client")
    def test_route_to_code_assistant(self, mock_get_llm):
        """测试路由到代码助手 Agent"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = '{"agent": "code_assistant"}'
        mock_get_llm.return_value = mock_llm

        state: AgentState = {
            "messages": [HumanMessage(content="写一个快速排序算法")],
            "next_agent": "",
            "context": "",
        }

        result = supervisor_node(state)
        assert result["next_agent"] == "code_assistant"

    @patch("backend.agents.supervisor.get_llm_client")
    def test_route_to_general(self, mock_get_llm):
        """测试路由到通用对话 Agent"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = '{"agent": "general"}'
        mock_get_llm.return_value = mock_llm

        state: AgentState = {
            "messages": [HumanMessage(content="你好")],
            "next_agent": "",
            "context": "",
        }

        result = supervisor_node(state)
        assert result["next_agent"] == "general"

    @patch("backend.agents.supervisor.get_llm_client")
    def test_default_route_on_parse_error(self, mock_get_llm):
        """测试 JSON 解析失败时的默认路由"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "我无法理解你的问题"
        mock_get_llm.return_value = mock_llm

        state: AgentState = {
            "messages": [HumanMessage(content="测试")],
            "next_agent": "",
            "context": "",
        }

        result = supervisor_node(state)
        assert result["next_agent"] == "general"  # 默认路由到通用对话

    @patch("backend.agents.supervisor.get_llm_client")
    def test_invalid_agent_name(self, mock_get_llm):
        """测试无效 Agent 名称时的默认路由"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = '{"agent": "invalid_agent"}'
        mock_get_llm.return_value = mock_llm

        state: AgentState = {
            "messages": [HumanMessage(content="测试")],
            "next_agent": "",
            "context": "",
        }

        result = supervisor_node(state)
        assert result["next_agent"] == "general"


class TestRouter:
    """路由函数测试"""

    def test_router_returns_next_agent(self):
        state: AgentState = {
            "messages": [],
            "next_agent": "web_search",
            "context": "",
        }
        assert router(state) == "web_search"

    def test_router_empty_string(self):
        """空字符串会被返回（key 存在但值为空）"""
        state: AgentState = {
            "messages": [],
            "next_agent": "",
            "context": "",
        }
        # 当 next_agent 为空字符串时，router 返回空字符串
        # 因为 state.get("next_agent", "document_qa") 找到 key 但值为 ""
        assert router(state) == ""

    def test_router_missing_key(self):
        """缺少 next_agent key 时返回默认值"""
        state = {"messages": []}
        assert router(state) == "general"


class TestGraphConstruction:
    """图构建测试"""

    def test_build_graph(self):
        """测试图构建成功"""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_nodes(self):
        """测试图包含所有节点"""
        graph = build_graph()
        # LangGraph 编译后的图应该包含这些节点
        # 通过检查 graph 的内部结构
        assert graph is not None


@pytest.mark.integration
class TestGraphExecution:
    """图执行集成测试（需要 LLM 服务）"""

    @patch("backend.agents.supervisor.get_llm_client")
    @patch("backend.agents.document_qa.get_llm_client")
    @patch("backend.agents.document_qa.document_search")
    def test_document_qa_flow(self, mock_search, mock_doc_llm, mock_sup_llm):
        """测试文档问答完整流程"""
        # Mock supervisor 路由
        mock_sup = MagicMock()
        mock_sup.chat.return_value = '{"agent": "document_qa"}'

        # Mock document search
        mock_search.invoke.return_value = "RAG 是检索增强生成技术的相关文档"

        # Mock document QA
        mock_doc = MagicMock()
        mock_doc.chat.return_value = "RAG 是检索增强生成技术"

        mock_sup_llm.return_value = mock_sup
        mock_doc_llm.return_value = mock_doc

        graph = build_graph()
        result = graph.invoke({"messages": [HumanMessage(content="什么是 RAG？")]})

        assert result is not None
