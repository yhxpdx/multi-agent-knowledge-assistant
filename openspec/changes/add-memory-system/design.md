## Context

当前系统存在两层"记忆断层"：

1. **短时记忆断层**：`chat.py` 中已从 Redis 获取对话历史（`history = redis.get_history_for_llm(session_id)`），但从未传入 agent graph。LLM 每次只看到当前一条用户消息，无法感知同一会话中的上下文。
2. **长期记忆缺失**：会话之间完全隔离，Redis TTL 7 天后数据自动清除。用户偏好、重要结论、知识积累等无法跨会话保留。

此外，各 agent 对历史消息的处理不统一：`document_qa`/`code_assistant`/`general` 读 `state["messages"]` 最近 6 条，`web_search` 完全不读历史。而 `state["messages"]` 通常只有 1 条 HumanMessage，这些 last-6 逻辑形同虚设。

现有基础设施：Milvus（向量库，已有 documents collection）、Redis（会话存储）、Ollama bge-m3（嵌入模型）。

## Goals / Non-Goals

**Goals:**
- 修复短时记忆断层：Redis 对话历史正确注入 agent graph，LLM 能感知会话上下文
- 实现长期记忆：跨会话存储和检索用户关键信息（偏好、结论、知识）
- 统一各 agent 的上下文获取方式
- 记忆系统的增删查可观测（API + 前端）

**Non-Goals:**
- 不做用户认证/多用户隔离（超出当前范围）
- 不做记忆的自动过期/遗忘曲线（简化实现，用 TTL 即可）
- 不做多 agent 间的 handoff/循环（保持当前单次路由架构）
- 不做记忆的版本管理/回滚

## Decisions

### Decision 1: 长期记忆存储选型 — Milvus 向量库

**选择**：复用现有 Milvus，新建 `memory_store` collection

**备选方案对比**：

| 方案 | 语义检索 | 实现复杂度 | 依赖 | 面试亮点 |
|------|---------|-----------|------|---------|
| **A. Milvus 向量库** | ✅ 支持 | 低（复用现有） | 无新增 | 高（RAG + Memory 统一架构） |
| B. Redis JSON | ❌ 仅关键词 | 低 | 无新增 | 低 |
| C. Redis + Milvus 混合 | ✅ 支持 | 高（双写一致性） | 无新增 | 中 |
| D. SQLite/文件 | ❌ 仅关键词 | 低 | 新增依赖 | 低 |

**为什么选 A**：
- 语义检索是记忆系统的核心能力。用户说"我之前问过那个排序算法"，关键词匹配找不到，但向量相似度可以
- 复用现有 Milvus + Ollama bge-m3，零新增依赖
- 架构统一：文档 RAG 和记忆检索共享同一套向量搜索基础设施
- 面试加分：能讲清楚"为什么记忆也要用向量检索而不是关键词匹配"

### Decision 2: 记忆提取策略 — LLM 判断 + 结构化输出

**选择**：每次对话结束后，由 LLM 判断是否值得提取记忆，输出结构化 JSON

**备选方案对比**：

| 方案 | 记忆质量 | API 调用次数 | 复杂度 |
|------|---------|------------|--------|
| A. 每轮都提取 | 低（噪声多） | 每轮 +1 | 低 |
| **B. LLM 判断是否提取** | 高（有选择） | 按需 +1 | 中 |
| C. 会话结束后批量提取 | 中（可能遗漏） | 每 session +1 | 中 |
| D. 基于规则提取 | 低（死板） | 0 | 低 |

**为什么选 B**：
- LLM 能理解语义：区分"用户说'你好'"（不存）和"用户喜欢 Python 胜过 Java"（存）
- 结构化输出可控：`{"should_save": true, "memory": "用户偏好 Python", "category": "preference"}`
- 只增加一次额外 LLM 调用（在对话回复之后），开销可控
- 避免存储大量噪声数据

### Decision 3: 记忆注入方式 — 通过 AgentState.context 字段

**选择**：启用已定义但未使用的 `context` 字段，Supervisor 在路由前检索记忆，写入 context

**备选方案对比**：

| 方案 | 改动量 | 一致性 | 扩展性 |
|------|--------|--------|--------|
| A. 每个 agent 独立检索 | 大（改4个文件） | 差（可能遗漏） | 差 |
| **B. Supervisor 统一注入 context** | 小（改2个文件） | 好（必经之路） | 好 |
| C. 新增 memory agent | 中（新增文件） | 好 | 过度设计 |

**为什么选 B**：
- Supervisor 是唯一入口，所有请求必经，统一检索保证一致性
- `context` 字段已定义但未使用，正好激活
- 各 agent 只需从 `state["context"]` 读取，无需各自调用 memory_manager
- 面试话术："Supervisor 不仅做路由，还负责上下文编排"

### Decision 4: 短时记忆修复 — 注入 Redis 历史到 graph state

**选择**：在 `chat.py` 构建 graph 输入时，将 Redis 历史转为 BaseMessage 列表传入

**为什么这样设计**：
- 最小改动：只需修改 `chat.py` 一处
- LangGraph 的 `add_messages` reducer 会自动合并历史和新消息
- 各 agent 已有的 `state["messages"]` 读取逻辑立刻生效
- 修复后：document_qa 的"最近 6 条"窗口就能真正看到对话历史

### Decision 5: memory_store 的 Collection 设计

```python
Collection: memory_store
Fields:
  - id: VARCHAR(64)       # UUID, 主键
  - content: VARCHAR(512) # 记忆内容
  - category: VARCHAR(32) # 分类: preference / fact / conclusion / instruction
  - session_id: VARCHAR(16) # 来源会话
  - created_at: FLOAT     # 创建时间戳
  - embedding: FLOAT(1024) # bge-m3 向量
Index: HNSW (M=16, efConstruction=256, metric=COSINE)
TTL: 90 天（长期但非永久）
```

**为什么单独建 collection 而不是和 documents 共存**：
- 生命周期不同：文档是永久的，记忆有 TTL
- 检索逻辑不同：文档检索 top_k=5，记忆检索 top_k=3
- 元数据不同：记忆有 category，文档有 doc_id/file_name
- 可独立管理：清空记忆不影响文档，反之亦然

## Risks / Trade-offs

- **[额外 LLM 调用开销]** → 记忆提取增加一次 LLM 调用。缓解：只在 LLM 判断 should_save=true 时才写入库；使用轻量级模型参数（temperature=0, max_tokens=200）
- **[记忆噪声累积]** → 无关紧要的信息可能被存入。缓解：category 分类 + 相似度去重（新记忆与已有记忆 cosine > 0.95 时不重复写入）
- **[记忆检索延迟]** → 每次对话多一次 Milvus 查询。缓解：记忆 collection 数据量小，HNSW 检索 <10ms；可在 supervisor 节点并行执行路由和记忆检索
- **[context 字段长度膨胀]** → 大量记忆注入可能导致 prompt 过长。缓解：限制 top_k=3，单条记忆 max 512 字符，总 context 上限约 1500 字符
- **[短时记忆修复的兼容性]** → 注入历史后 state["messages"] 变长，agent 的 last-6 窗口逻辑需要确认无副作用。缓解：add_messages reducer 保证顺序正确
