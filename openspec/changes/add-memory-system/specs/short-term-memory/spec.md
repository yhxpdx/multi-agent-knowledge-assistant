## ADDED Requirements

### Requirement: Redis 对话历史注入 agent graph
系统 SHALL 在调用 agent graph 时将 Redis 中的对话历史转换为 BaseMessage 列表，作为 graph 输入的一部分传入，使 LLM 能感知同一会话内的上下文。

#### Scenario: 首条消息无历史
- **WHEN** 用户在新会话中发送第一条消息
- **THEN** graph 输入仅包含当前 HumanMessage，不注入历史

#### Scenario: 多轮对话历史注入
- **WHEN** 用户在已有历史的会话中发送消息
- **THEN** 系统从 Redis 获取该会话的历史消息，转换为 HumanMessage/AIMessage 列表，与当前消息一起传入 graph

#### Scenario: 历史消息截断
- **WHEN** 会话历史超过 20 条消息
- **THEN** 系统仅注入最近 20 条消息，避免 context 过长

### Requirement: 统一各 agent 的上下文读取方式
所有 agent SHALL 从 `state["messages"]` 读取对话历史，从 `state["context"]` 读取长期记忆上下文，不再各自独立处理历史。

#### Scenario: agent 读取对话历史
- **WHEN** 任意 agent 被调度执行
- **THEN** 该 agent 从 `state["messages"]` 获取对话历史（已包含 Redis 注入的历史消息）

#### Scenario: web_search_agent 修复
- **WHEN** web_search_agent 被调度执行
- **THEN** 该 agent SHALL 读取 state["messages"] 中的最近对话历史，与其他 agent 行为一致

#### Scenario: agent 读取长期记忆上下文
- **WHEN** 任意 agent 被调度执行且 state["context"] 非空
- **THEN** 该 agent SHALL 将 context 内容作为补充上下文注入系统提示

### Requirement: Supervisor 统一上下文编排
Supervisor 节点 SHALL 在路由决策的同时，检索相关长期记忆并写入 state["context"]，使下游 agent 统一获取上下文。

#### Scenario: 检索到相关记忆
- **WHEN** 用户消息与长期记忆库中的记忆语义相关
- **THEN** Supervisor 将检索到的记忆格式化后写入 state["context"]

#### Scenario: 无相关记忆
- **WHEN** 用户消息与长期记忆库中的记忆无语义相关
- **THEN** Supervisor 将 state["context"] 设为空字符串
