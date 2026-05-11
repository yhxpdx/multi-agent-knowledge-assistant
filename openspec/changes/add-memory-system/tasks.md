## 1. 短时记忆修复

- [x] 1.1 修改 chat.py：将 Redis 对话历史转为 BaseMessage 列表注入 agent graph 输入
- [x] 1.2 修改 chat.py：限制注入历史最多 20 条消息
- [x] 1.3 修改 web_search_agent.py：添加对话历史读取逻辑，与其他 agent 保持一致
- [x] 1.4 统一所有 agent 的 context 读取：从 state["context"] 获取长期记忆上下文并注入系统提示

## 2. 长期记忆核心模块

- [x] 2.1 创建 backend/core/memory_manager.py：MemoryManager 类（单例模式）
- [x] 2.2 实现 create_collection()：自动创建 memory_store collection（id/content/category/session_id/created_at/embedding 字段 + HNSW 索引）
- [x] 2.3 实现 add_memory()：存储单条记忆（embedding + 元数据），含相似度去重（>0.95 不写入）
- [x] 2.4 实现 search_memories()：根据查询向量检索 top-3 相关记忆，返回格式化结果
- [x] 2.5 实现 delete_memory()：按 ID 删除单条记忆
- [x] 2.6 实现 list_memories()：列出所有记忆，支持按 category 筛选
- [x] 2.7 实现 get_memory_manager()：全局单例获取函数

## 3. 记忆提取机制

- [x] 3.1 设计记忆提取 prompt（MEMORY_EXTRACT_PROMPT），输出结构化 JSON：should_save/memory/category
- [x] 3.2 实现 extract_memory()：调用 LLM 判断是否提取，解析 JSON，category 校验与默认值
- [x] 3.3 实现容错：LLM 调用失败或 JSON 解析错误时记录日志，不影响主流程
- [x] 3.4 在 chat.py 的对话回复后调用 extract_memory()，异步写入记忆库

## 4. Supervisor 上下文编排

- [x] 4.1 修改 supervisor_node()：在路由决策前，调用 memory_manager.search_memories() 检索相关长期记忆
- [x] 4.2 将检索到的记忆格式化写入 state["context"]
- [x] 4.3 无相关记忆时 context 设为空字符串

## 5. 记忆管理 API

- [x] 5.1 创建 backend/api/memories.py：GET /api/memories（列出所有记忆，支持 category 筛选）
- [x] 5.2 实现 DELETE /api/memories/{memory_id}（删除指定记忆，404 处理）
- [x] 5.3 在 main.py 注册 memories_router

## 6. 前端记忆展示

- [x] 6.1 在 Streamlit 侧边栏添加"记忆管理"区域：展示记忆列表
- [x] 6.2 支持按 category 筛选记忆
- [x] 6.3 支持删除单条记忆

## 7. 测试

- [x] 7.1 单元测试：test_memory_manager.py — collection 创建、增删查、去重、格式化
- [x] 7.2 单元测试：test_memory_extract.py — 提取 prompt、JSON 解析、category 校验、容错
- [x] 7.3 单元测试：test_short_term_memory.py — 历史注入、截断、web_search 修复
- [x] 7.4 集成测试：test_memory_flow.py — 完整流程（对话→提取→存储→检索→注入）
- [x] 7.5 API 测试：test_memories_api.py — 列出/删除/404/筛选

## 8. 文档与配置

- [x] 8.1 更新 .env.example：添加记忆相关配置项（MEMORY_TTL_DAYS 等）
- [x] 8.2 更新 README.md：记忆系统架构说明、使用方式
- [x] 8.3 更新 docs/project_evaluation.md：记忆系统评估
- [ ] 8.4 Git commit + GitHub push
