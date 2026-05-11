## ADDED Requirements

### Requirement: 长期记忆存储
系统 SHALL 在 Milvus 中维护独立的 `memory_store` collection，存储跨会话的关键信息，包括用户偏好、事实结论、指令约束等。

#### Scenario: 创建 memory_store collection
- **WHEN** 系统首次启动且 memory_store collection 不存在
- **THEN** 自动创建 collection，包含 id、content、category、session_id、created_at、embedding 字段，建立 HNSW 索引

#### Scenario: 记忆去重
- **WHEN** 新提取的记忆与已有记忆的余弦相似度 > 0.95
- **THEN** 系统不重复写入该记忆，保留已有版本

### Requirement: 记忆自动提取
系统 SHALL 在每次对话回复完成后，由 LLM 判断是否提取值得长期保存的信息，并结构化写入记忆库。

#### Scenario: 提取用户偏好
- **WHEN** 用户在对话中表达偏好（如"我喜欢用 Python"），LLM 判断 should_save=true
- **THEN** 系统将记忆以 category="preference" 写入 memory_store

#### Scenario: 提取重要结论
- **WHEN** 对话中产生重要结论（如"项目使用 HNSW 索引"），LLM 判断 should_save=true
- **THEN** 系统将记忆以 category="conclusion" 写入 memory_store

#### Scenario: 闲聊不提取
- **WHEN** 对话为闲聊（如"你好"、"今天天气不错"），LLM 判断 should_save=false
- **THEN** 系统不写入任何记忆

#### Scenario: 记忆提取失败容错
- **WHEN** 记忆提取的 LLM 调用失败或 JSON 解析出错
- **THEN** 系统记录警告日志，不影响主对话流程

### Requirement: 记忆语义检索
系统 SHALL 根据用户当前问题，从 memory_store 中语义检索相关记忆，供 agent 作为上下文使用。

#### Scenario: 检索到相关记忆
- **WHEN** 用户发送消息，且 memory_store 中存在语义相关的记忆
- **THEN** 系统返回 top-3 最相关记忆，附带相似度分数

#### Scenario: 记忆库为空
- **WHEN** 用户发送消息，但 memory_store 中无任何记忆
- **THEN** 系统返回空结果，不影响正常对话

#### Scenario: 检索结果格式化
- **WHEN** 检索到记忆
- **THEN** 每条记忆格式为 "[{category}] {content}"，多条记忆以换行分隔

### Requirement: 记忆管理 API
系统 SHALL 提供记忆管理的 REST API，支持查看和删除记忆。

#### Scenario: 列出所有记忆
- **WHEN** 调用 GET /api/memories
- **THEN** 返回所有记忆列表，每条包含 id、content、category、created_at、similarity_score

#### Scenario: 按分类筛选记忆
- **WHEN** 调用 GET /api/memories?category=preference
- **THEN** 仅返回 category 为 preference 的记忆

#### Scenario: 删除指定记忆
- **WHEN** 调用 DELETE /api/memories/{memory_id}
- **THEN** 从 memory_store 中删除该记忆，返回成功状态

#### Scenario: 删除不存在的记忆
- **WHEN** 调用 DELETE /api/memories/{nonexistent_id}
- **THEN** 返回 404 状态码

### Requirement: 记忆提取 prompt 设计
记忆提取 SHALL 使用专门的系统提示，引导 LLM 输出结构化 JSON，判断是否保存及保存内容。

#### Scenario: 提取 prompt 输出格式
- **WHEN** LLM 进行记忆提取判断
- **THEN** 输出格式为 `{"should_save": boolean, "memory": string, "category": "preference|fact|conclusion|instruction"}`

#### Scenario: category 限制
- **WHEN** LLM 输出的 category 不在预定义列表中
- **THEN** 系统将该 category 默认设为 "fact"
