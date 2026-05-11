# Multi-Agent Knowledge Assistant (多智能体知识助手)

> Python 3.11+ | LangGraph | FastAPI | Milvus | Redis | Streamlit

基于 LangGraph 的多智能体知识助手系统，展示 RAG + 多智能体协作 + 工具调用全链路能力。

## 架构设计

```
用户 → Streamlit 前端 → FastAPI 后端
                            ↓
                     LangGraph Supervisor
                      ↙       ↓       ↘
            DocQA Agent  WebSearch Agent  Code Agent
                ↓              ↓             ↓
           Milvus RAG    DuckDuckGo    沙箱执行
                ↓
              Redis (对话记忆)
```

## 技术栈

| 组件 | 技术 | 选型理由 |
|------|------|---------|
| Agent 框架 | LangGraph | 图状态机，支持 Supervisor 模式，2025 年最主流 |
| LLM | 火山引擎 Doubao-Seed-2.0-lite | OpenAI 兼容接口，中文优化 |
| 向量数据库 | Milvus | 企业级，HNSW 索引，已 Docker 部署 |
| Embedding | bge-m3 (Ollama) | Recall@5 100%，中英文均衡 |
| 对话记忆 | Redis | 微秒级读写，TTL 自动过期 |
| 后端 | FastAPI | 异步支持，SSE 流式响应 |
| 前端 | Streamlit | Python 全栈，内置聊天组件 |

## 核心功能

### 1. 多智能体协作 (Supervisor 模式)
- **Supervisor Agent**: 意图识别 + 任务路由
- **Document QA Agent**: RAG 检索 + 文档问答
- **Web Search Agent**: 联网搜索 + 信息总结
- **Code Assistant Agent**: 代码生成/解释 + 沙箱执行

### 2. RAG 知识库
- 支持 PDF/TXT/Markdown/DOCX 文档上传
- 自动解析 → 分块 → Embedding → Milvus 存储
- 向量相似度搜索 + 来源引用

### 3. 工具调用 (Function Calling)
- `document_search`: 知识库检索
- `web_search`: DuckDuckGo 联网搜索
- `code_executor`: 受限 Python 代码执行
- `calculator`: 数学计算

### 4. 对话记忆
- Redis 存储对话历史
- 按 session 隔离
- 7 天 TTL 自动过期

## 快速开始

### 前置条件
- Python 3.11+
- Docker (Milvus, Redis, Ollama)

### 启动服务

```bash
# 1. 启动依赖服务 (Docker)
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest
docker run -d --name redis -p 6379:6379 redis:latest
docker run -d --name ollama -p 11434:11434 ollama/ollama

# 2. 拉取 Embedding 模型
docker exec ollama ollama pull bge-m3

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 5. 导入知识库数据
python data/ingest_data.py

# 6. 启动后端
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 7. 启动前端 (另一个终端)
streamlit run frontend/app.py
```

### Docker Compose 一键启动

```bash
docker-compose up -d
```

## 项目结构

```
multi-agent-knowledge-assistant/
├── backend/
│   ├── agents/          # LangGraph 智能体
│   │   ├── supervisor.py      # Supervisor 路由
│   │   ├── document_qa.py     # 文档问答 Agent
│   │   ├── web_search_agent.py # 联网搜索 Agent
│   │   └── code_assistant.py  # 代码助手 Agent
│   ├── api/             # FastAPI 路由
│   │   ├── chat.py            # 聊天 API (SSE)
│   │   ├── documents.py       # 文档管理 API
│   │   ├── sessions.py        # 会话管理 API
│   │   └── health.py          # 健康检查
│   ├── core/            # 核心服务
│   │   ├── config.py          # 配置管理
│   │   ├── embedding.py       # Embedding 服务
│   │   ├── llm_client.py      # LLM 客户端
│   │   ├── milvus_client.py   # Milvus 客户端
│   │   ├── redis_manager.py   # Redis 管理
│   │   └── document_parser.py # 文档解析
│   ├── tools/           # 工具
│   │   ├── document_search.py
│   │   ├── web_search.py
│   │   ├── code_executor.py
│   │   └── calculator.py
│   └── main.py          # FastAPI 入口
├── frontend/
│   └── app.py           # Streamlit 前端
├── data/
│   ├── raw/             # 原始数据
│   ├── collect_data.py  # 数据采集脚本
│   └── ingest_data.py   # 数据导入脚本
├── evals/               # 评估脚本
│   ├── eval_embedding.py
│   └── eval_chunking.py
├── docker-compose.yml
├── requirements.txt
└── .env
```

## API 文档

启动后端后访问 http://localhost:8000/docs 查看 Swagger 文档。

### 主要端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/chat | 聊天 (支持 SSE 流式) |
| POST | /api/documents | 上传文档 |
| GET | /api/documents | 文档列表 |
| DELETE | /api/documents/{doc_id} | 删除文档 |
| POST | /api/sessions | 创建会话 |
| GET | /api/sessions | 会话列表 |
| GET | /api/health | 健康检查 |

## 技术选型评估报告

### Embedding 模型评估

| 指标 | nomic-embed-text | bge-m3 |
|------|-----------------|--------|
| 向量维度 | 768 | 1024 |
| Recall@5 | 44.44% | **100.00%** |
| Recall@5 (中文) | 66.67% | **100.00%** |
| Recall@5 (英文) | 0.00% | **100.00%** |
| MRR | 29.81% | **97.22%** |
| 平均查询耗时 | 0.091s | 0.375s |

**结论**: 选择 bge-m3，检索准确率碾压，中英文均衡。

### 分块策略评估

| chunk_size | overlap | chunks | Recall@5 | MRR |
|-----------|---------|--------|----------|-----|
| 200 | 20 | 818 | 100% | 100% |
| **500** | **50** | **318** | **100%** | **100%** |
| 800 | 80 | 191 | 100% | 100% |

**结论**: 选择 chunk_size=500, overlap=50，平衡存储和检索质量。

### Agent 架构选型

选择 **Supervisor 模式**:
- 面试高频考点
- 职责清晰，易于扩展
- 代码量适中，适合简历项目展示
