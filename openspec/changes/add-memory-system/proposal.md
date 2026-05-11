## Why

当前系统的"记忆"存在严重断层：虽然 Redis 存储了会话历史，但 `chat.py` 中获取的对话历史从未传入 agent graph，LLM 每次只看到当前一条用户消息，完全没有上下文。此外，会话之间完全隔离，没有跨会话的长期记忆（用户偏好、知识积累等），导致用户每次开启新会话都要从零开始。这直接影响了系统的智能程度和用户体验，也是面试中高频被追问的架构短板。

## What Changes

- **修复短时记忆断层**：将 Redis 中已有的对话历史正确注入 agent graph，使 LLM 能感知同一会话内的上下文
- **新增长期记忆系统**：基于 Milvus 向量库存储跨会话的关键信息（用户偏好、重要结论、知识摘要等），支持语义检索
- **新增记忆提取与写入机制**：在对话结束后，由 LLM 自动提取值得长期保存的信息，写入向量记忆库
- **新增记忆检索与注入机制**：在新会话中，根据用户问题检索相关长期记忆，注入 agent 的上下文
- **统一各 Agent 的历史处理**：修复 web_search_agent 不传历史的 bug，统一所有 agent 的上下文注入方式
- **更新 Supervisor 路由**：利用 `context` 字段传递检索到的长期记忆，让所有 agent 都能访问

## Capabilities

### New Capabilities
- `short-term-memory`: 修复 Redis 对话历史注入 agent graph 的断层问题，统一所有 agent 的上下文获取方式
- `long-term-memory`: 基于 Milvus 的跨会话长期记忆系统，包括记忆提取（对话→向量）、记忆检索（问题→相关记忆）、记忆注入（检索结果→agent 上下文）

### Modified Capabilities

## Impact

- **核心代码变更**：`chat.py`（历史注入）、`supervisor.py`（context 字段启用）、所有 agent 文件（统一上下文读取方式）
- **新增模块**：`backend/core/memory_manager.py`（长期记忆管理器）
- **新增 Milvus Collection**：`memory_store`（独立于文档库，存储记忆向量）
- **新增 API**：记忆管理相关端点（查看/删除记忆）
- **配置变更**：`.env` 新增记忆相关参数（TTL、提取阈值等）
- **依赖**：无需新增外部依赖，复用现有 Milvus + Redis + Ollama 基础设施
