"""
端到端测试 — 验证多智能体路由准确率和回答质量

测试目标：
1. 验证 Supervisor 路由准确性
2. 验证 RAG 回答质量
3. 验证各 Agent 功能正常
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from backend.agents.supervisor import agent_graph


def test_routing():
    """测试 Supervisor 路由准确性"""
    test_cases = [
        {"query": "什么是RAG技术？", "expected_agent": "document_qa"},
        {"query": "今天有什么新闻？", "expected_agent": "web_search"},
        {"query": "写一个排序算法", "expected_agent": "code_assistant"},
        {"query": "解释一下LangGraph", "expected_agent": "document_qa"},
        {"query": "计算 2 的 10 次方", "expected_agent": "code_assistant"},
        {"query": "搜索最新的AI论文", "expected_agent": "web_search"},
    ]

    print("=" * 60)
    print("路由准确率测试")
    print("=" * 60)

    correct = 0
    total = len(test_cases)

    for tc in test_cases:
        result = agent_graph.invoke({"messages": [HumanMessage(content=tc["query"])]})
        actual_agent = result.get("next_agent", "unknown")
        expected = tc["expected_agent"]
        hit = actual_agent == expected
        if hit:
            correct += 1

        status = "PASS" if hit else "FAIL"
        print(f"  [{status}] \"{tc['query']}\"")
        print(f"         Expected: {expected}, Got: {actual_agent}")

    accuracy = correct / total
    print(f"\n路由准确率: {correct}/{total} = {accuracy:.2%}")
    return accuracy


def test_rag_quality():
    """测试 RAG 回答质量"""
    test_cases = [
        {
            "query": "What is RAG?",
            "check_keywords": ["Retrieval", "retrieval", "检索", "Generation"],
        },
        {
            "query": "What are the components of an AI Agent?",
            "check_keywords": ["Planning", "Memory", "Tool", "Agent"],
        },
    ]

    print("\n" + "=" * 60)
    print("RAG 回答质量测试")
    print("=" * 60)

    passed = 0
    total = len(test_cases)

    for tc in test_cases:
        result = agent_graph.invoke({"messages": [HumanMessage(content=tc["query"])]})
        response = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.type == "ai":
                response = msg.content
                break

        # 检查是否包含关键词
        hits = [kw for kw in tc["check_keywords"] if kw in response]
        hit = len(hits) > 0
        if hit:
            passed += 1

        status = "PASS" if hit else "FAIL"
        print(f"  [{status}] \"{tc['query']}\"")
        print(f"         Keywords found: {hits}")
        print(f"         Response preview: {response[:100]}...")

    quality = passed / total
    print(f"\nRAG 质量: {passed}/{total} = {quality:.2%}")
    return quality


def main():
    print("=" * 60)
    print("端到端测试开始")
    print("=" * 60)

    routing_accuracy = test_routing()
    rag_quality = test_rag_quality()

    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    print(f"  路由准确率: {routing_accuracy:.2%}")
    print(f"  RAG 质量: {rag_quality:.2%}")

    if routing_accuracy >= 0.8 and rag_quality >= 0.8:
        print("\n所有测试通过！")
    else:
        print("\n部分测试未通过，需要检查。")


if __name__ == "__main__":
    main()
