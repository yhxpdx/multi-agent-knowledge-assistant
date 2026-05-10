"""
测试配置和共享 fixtures

全链路思考 — 测试策略：
1. 单元测试：每个模块独立测试，mock 外部依赖
2. 集成测试：模块间交互测试，使用真实服务
3. API 测试：HTTP 接口测试，验证请求/响应格式
4. 端到端测试：完整流程测试，从用户输入到最终输出
"""

import sys
import os
import pytest

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_text():
    """示例文本，用于分块和 embedding 测试"""
    return """
    RAG（Retrieval-Augmented Generation）是一种结合信息检索和文本生成的技术范式。
    RAG 的工作流程分为三个阶段：
    1. 索引阶段（Indexing）：将文档进行分块（Chunking），通过 Embedding 模型转换为向量，存储到向量数据库中。
    2. 检索阶段（Retrieval）：将用户查询同样转换为向量，在向量数据库中执行相似度搜索，找到最相关的文档片段。
    3. 生成阶段（Generation）：将检索到的文档片段作为上下文，与用户问题一起输入 LLM，生成最终回答。
    """


@pytest.fixture
def sample_documents():
    """示例文档列表"""
    return [
        {
            "title": "RAG 技术详解",
            "content": "RAG 是一种结合检索和生成的技术。它通过从知识库中检索相关文档，然后将这些文档作为上下文输入 LLM。",
            "source": "test",
        },
        {
            "title": "Agent 架构",
            "content": "AI Agent 是一种能够自主感知环境、做出决策并执行行动的智能系统。核心模块包括规划、记忆、工具使用和反思。",
            "source": "test",
        },
    ]


@pytest.fixture
def test_queries():
    """测试查询集"""
    return [
        {"query": "什么是 RAG？", "expected_doc": "RAG 技术详解"},
        {"query": "Agent 的核心模块有哪些？", "expected_doc": "Agent 架构"},
    ]
