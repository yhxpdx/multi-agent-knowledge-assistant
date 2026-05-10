# 项目亮点总结 — 简历用

## 项目名称
多智能体知识助手 (Multi-Agent Knowledge Assistant)

## 一句话描述
基于 LangGraph + RAG + Milvus 的多智能体知识助手系统，支持文档问答、联网搜索、代码生成三大功能。

## 核心亮点

### 1. 多智能体协作架构
- 使用 LangGraph Supervisor 模式实现任务自动路由
- 三个专业子 Agent：文档问答、联网搜索、代码助手
- Supervisor 通过 LLM 意图识别，准确路由到对应 Agent
- 支持 SSE 流式输出，实时展示 Agent 执行过程

### 2. 完整 RAG 管道
- 文档上传 → 解析 → 分块 → Embedding → Milvus 向量存储
- 支持 PDF/TXT/Markdown/DOCX 多种格式
- 基于 bge-m3 的 1024 维向量检索，Recall@5 达到 100%
- 检索结果带来源引用，减少幻觉

### 3. 数据驱动的技术选型
- **Embedding 模型评估**：用 18 个测试查询对比 nomic-embed-text 和 bge-m3，bge-m3 Recall@5 100% vs 44%
- **分块策略评估**：对比 7 种参数组合，确定 chunk_size=500, overlap=50
- **向量数据库选型**：Milvus + HNSW 索引，适合中小规模生产环境
- 每个决策都有数据支撑和对比分析

### 4. 工具调用 (Function Calling)
- 实现 4 个工具：文档搜索、联网搜索、代码执行、计算器
- 代码执行工具实现安全沙箱：模块黑名单、函数黑名单、输出限制
- 工具描述精确，LLM 选择准确率高

### 5. 生产级工程实践
- FastAPI 后端，支持异步和 SSE 流式响应
- Redis 对话记忆管理，7 天 TTL 自动过期
- Docker Compose 一键部署
- 完整的健康检查和错误处理

## 技术栈
LangGraph, LangChain, Milvus, Redis, FastAPI, Streamlit, Ollama, bge-m3, Doubao LLM

## 可量化的成果
- Embedding 模型评估：bge-m3 Recall@5 100%，比 nomic-embed-text 提升 125%
- 分块策略：500/50 参数下 318 个 chunks，检索延迟 <100ms
- 多智能体路由：三种任务类型准确路由
- 知识库：25 篇文档，299 个 chunks，覆盖 AI/Agent/RAG 领域

## 面试话术

> 这个项目展示了我从 0 到 1 构建 Agent 应用的能力。我不是简单地调用 API，而是做了完整的技术选型评估——比如 Embedding 模型，我用实际数据对比了两个模型，bge-m3 在我的数据上 Recall@5 达到 100%，而 nomic-embed-text 只有 44%，这个数据驱动的决策过程是项目的核心亮点。
>
> 在架构上，我选择 LangGraph Supervisor 模式，因为它职责清晰、易于扩展。Supervisor 通过 LLM 意图识别自动路由到三个专业子 Agent，每个子 Agent 有自己的工具集。
>
> 在工程上，我注重生产级实践：FastAPI 异步支持、SSE 流式输出、Redis 对话记忆、Docker 部署。这些都体现了我把 Agent 从 demo 做到生产级的能力。
