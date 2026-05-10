"""
数据采集脚本 — 从公开来源采集 AI/Agent/RAG 相关技术文档

数据选型理由（全链路思考）:
1. 为什么选 LangChain 官方文档？
   - 与项目技术栈直接相关，面试时可以讲"知识库内容和项目技术栈一致"
   - 文档结构化程度高（标题、段落、代码块），适合测试分块策略
   - 中英文混合，可以测试 Embedding 模型的多语言能力

2. 为什么选中文 AI 技术文章？
   - 目标岗位在国内，中文检索是核心需求
   - 内容涵盖 RAG、Agent、Prompt Engineering 等面试高频话题
   - 可以验证中文分块和检索的效果

3. 数据规模考量：
   - 不追求海量（百万级），而是追求领域相关性
   - 几百篇高质量技术文档足以展示 RAG 管道能力
   - 面试时可以解释"为什么选择领域相关数据而非通用语料"
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / "raw"
DATA_DIR.mkdir(exist_ok=True)


def collect_langchain_docs():
    """
    采集 LangChain 官方文档的关键页面
    选择这些页面的理由：
    - agents: Agent 核心概念，面试必问
    - rag: RAG 管道，项目核心功能
    - tools: 工具调用，Agent 的关键能力
    - langgraph: 多智能体编排，项目架构基础
    """
    base_url = "https://python.langchain.com/docs"
    pages = [
        # Agent 相关
        "/concepts/agents",
        "/concepts/chat_models",
        "/concepts/function_calling",
        "/concepts/messages",
        # RAG 相关
        "/concepts/rag",
        "/concepts/retrievers",
        "/concepts/vectorstores",
        "/concepts/embedding_models",
        "/concepts/text_splitters",
        "/concepts/document_loaders",
        "/concepts/output_parsers",
        # LangGraph 相关
        "/concepts/langgraph",
        "/concepts/low_level",
        # 工具相关
        "/concepts/tools",
        "/concepts/tool_calling",
    ]

    docs = []
    for page_path in pages:
        url = f"{base_url}{page_path}"
        try:
            resp = requests.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (research bot)"
            })
            if resp.status_code == 200:
                # 简单提取文本内容（去除 HTML 标签）
                text = extract_text_from_html(resp.text)
                if len(text) > 100:  # 过滤空页面
                    doc = {
                        "source": "langchain_docs",
                        "url": url,
                        "title": page_path.split("/")[-1].replace("-", " ").title(),
                        "content": text,
                        "collected_at": datetime.now().isoformat(),
                    }
                    docs.append(doc)
                    print(f"  [OK] {page_path} ({len(text)} chars)")
                else:
                    print(f"  [SKIP] {page_path} (too short)")
            else:
                print(f"  [FAIL] {page_path} (HTTP {resp.status_code})")
            time.sleep(1)  # 礼貌性延迟
        except Exception as e:
            print(f"  [ERROR] {page_path}: {e}")

    return docs


def collect_ai_knowledge_articles():
    """
    生成 AI/Agent/RAG 领域的技术知识文档
    这些内容基于公开的技术概念，用于构建领域知识库

    为什么自己生成而不是只爬取？
    - 保证内容质量和一致性
    - 覆盖面试高频知识点
    - 可以控制数据分布（确保各主题均衡）
    """
    articles = [
        {
            "title": "RAG 检索增强生成技术详解",
            "content": """RAG（Retrieval-Augmented Generation）是一种结合信息检索和文本生成的技术范式。

核心原理：
RAG 的工作流程分为三个阶段：
1. 索引阶段（Indexing）：将文档进行分块（Chunking），通过 Embedding 模型转换为向量，存储到向量数据库中。
2. 检索阶段（Retrieval）：将用户查询同样转换为向量，在向量数据库中执行相似度搜索，找到最相关的文档片段。
3. 生成阶段（Generation）：将检索到的文档片段作为上下文，与用户问题一起输入 LLM，生成最终回答。

RAG 的三种范式：
- Naive RAG：最基本的检索-生成流程，适合简单场景
- Advanced RAG：在 Naive 基础上增加预检索（查询重写、扩展）和后检索（重排序、压缩）优化
- Modular RAG：模块化架构，支持自定义组件组合，适合复杂生产环境

分块策略：
- 固定大小分块（Fixed-size Chunking）：按字符数分割，简单但可能切断语义
- 递归字符分割（Recursive Character Splitting）：按分隔符层级递归分割，保持语义完整性
- 语义分块（Semantic Chunking）：基于语义相似度动态确定分割点，效果最好但计算开销大
- 文档结构分块：按文档结构（标题、段落）分割，适合结构化文档

常见问题与优化：
- 幻觉问题（Hallucination）：LLM 生成了检索内容中不存在的信息。解决方案：增加引用验证、设置置信度阈值
- 检索噪声：检索到了不相关的内容干扰生成。解决方案：重排序（Reranking）、上下文压缩
- 多跳推理：问题需要综合多个文档的信息。解决方案：迭代检索、查询分解
""",
        },
        {
            "title": "AI Agent 核心架构与设计模式",
            "content": """AI Agent 是一种能够自主感知环境、做出决策并执行行动的智能系统。

Agent 的四大核心模块：
1. 规划（Planning）：将复杂任务分解为可执行的子任务序列
   - ReAct（Reasoning + Acting）：交替进行推理和行动
   - Plan-and-Execute：先制定完整计划，再逐步执行
   - Tree of Thoughts：探索多条推理路径，选择最优方案
   - Reflexion：通过自我反思改进决策质量

2. 记忆（Memory）：
   - 短期记忆：当前对话的上下文信息
   - 长期记忆：持久化存储的历史知识（通常用向量数据库）
   - 情景记忆：特定事件的详细记录

3. 工具使用（Tool Use）：
   - Function Calling：LLM 生成结构化的函数调用请求
   - 工具注册：定义工具的名称、描述、参数 schema
   - 工具执行：实际调用外部 API 或本地函数
   - 结果处理：将工具返回结果整合到对话上下文

4. 反思（Reflection）：
   - 自我评估：检查输出质量
   - 错误修正：识别并纠正错误
   - 经验积累：从过往任务中学习

多智能体协作模式：
- Supervisor（监督者）模式：一个主 Agent 负责任务分配，子 Agent 负责执行
- Swarm（群体）模式：Agent 之间直接通信和任务传递
- 层级模式：多层 Agent 组织，逐级细化任务
- 辩论模式：多个 Agent 对同一问题提出不同观点，最终达成共识

LangGraph 架构：
LangGraph 是 LangChain 团队开发的图状态机框架，核心概念：
- StateGraph：定义状态和转换规则
- Node：处理函数，执行具体逻辑
- Edge：节点间的连接，支持条件路由
- State：贯穿整个图执行的共享状态
- Checkpoint：状态持久化，支持断点恢复
""",
        },
        {
            "title": "向量数据库技术选型与实践",
            "content": """向量数据库是 RAG 系统的核心基础设施，负责高效存储和检索高维向量。

主流向量数据库对比：
1. Milvus
   - 架构：分布式，支持水平扩展
   - 索引类型：IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW, ANNOY 等
   - 适用场景：大规模生产环境，需要高可用和高性能
   - 部署方式：Docker、Kubernetes、Milvus Cloud
   - 优势：社区活跃（32K+ stars），企业级特性完善

2. ChromaDB
   - 架构：轻量级，嵌入式
   - 适用场景：原型开发、小规模应用
   - 优势：API 简洁，上手快
   - 劣势：不支持分布式，大规模性能受限

3. FAISS（Facebook AI Similarity Search）
   - 架构：库而非数据库，纯内存
   - 适用场景：高性能本地搜索，大规模离线处理
   - 优势：速度极快，支持 GPU 加速
   - 劣势：无持久化，需要自行管理

4. Qdrant
   - 架构：Rust 实现，高性能
   - 适用场景：需要高性能和丰富过滤的场景
   - 优势：内存效率高，支持复杂的 payload 过滤

选型决策框架：
选择向量数据库时需要考虑的关键维度：
- 数据规模：小（<10万）→ ChromaDB；中（10-1000万）→ Milvus/Qdrant；大（>1000万）→ Milvus 分布式
- 部署复杂度：开发阶段优先 ChromaDB；生产环境优先 Milvus
- 查询延迟：P99 延迟要求 <10ms → FAISS/Milvus HNSW
- 过滤能力：需要复杂元数据过滤 → Qdrant/Milvus
- 社区生态：Milvus > FAISS > ChromaDB > Qdrant

为什么本项目选择 Milvus？
1. 已有 Docker 部署，零额外部署成本
2. 项目定位是展示生产级能力，Milvus 是企业首选
3. 支持丰富的索引类型，可以做性能对比实验
4. Python SDK 完善，与 LangChain 集成好
""",
        },
        {
            "title": "Embedding 模型选型与评估方法",
            "content": """Embedding 模型将文本转换为稠密向量，是 RAG 系统质量的关键决定因素。

Embedding 模型的关键评估维度：
1. 向量维度：影响存储成本和检索速度
   - 低维（384-768）：存储小，速度快，但表达能力有限
   - 高维（1024-4096）：表达能力强，但存储和计算成本高

2. 检索准确率：核心指标
   - Recall@K：前 K 个结果中包含正确答案的比例
   - MRR（Mean Reciprocal Rank）：正确答案排名的倒数均值
   - NDCG：考虑排名位置的评分

3. 多语言支持：
   - 英文模型（如 text-embedding-ada-002）：英文效果好，中文一般
   - 中文优化模型（如 BGE、M3E）：中文效果好
   - 多语言模型（如 multilingual-e5-large）：中英文均衡

4. 推理速度：
   - API 模型：受网络延迟影响，通常 50-200ms
   - 本地模型：取决于硬件，CPU 100-500ms，GPU 5-50ms

常见 Embedding 模型对比：
- OpenAI text-embedding-3-small：1536 维，英文为主，API 调用
- BGE-large-zh-v1.5：1024 维，中文优化，本地部署
- nomic-embed-text：768 维，轻量级，本地部署（Ollama）
- BGE-M3：1024 维，多语言，支持稠密+稀疏+多向量检索
- M3E-base：768 维，中文优化，开源

评估方法论：
1. 准备测试集：包含 (query, positive_doc, negative_docs) 三元组
2. 计算向量：分别对 query 和 documents 计算 embedding
3. 执行检索：对每个 query，从所有 documents 中检索 Top-K
4. 计算指标：Recall@K, MRR, NDCG
5. 对比分析：不同模型在同一测试集上的表现

本项目的评估计划：
- 测试集：从知识库中抽取 50 个问题，每个问题关联 3 个相关文档
- 候选模型：nomic-embed-text（本地 768 维）、bge-m3（本地 1024 维）
- 评估维度：Recall@5、平均检索延迟、中文 vs 英文查询表现
""",
        },
        {
            "title": "Prompt Engineering 最佳实践",
            "content": """Prompt Engineering 是设计和优化 LLM 输入提示的技术，直接影响 Agent 的输出质量。

核心原则：
1. 明确性（Clarity）：指令要清晰明确，避免歧义
2. 具体性（Specificity）：给出具体的输出格式和约束
3. 结构化（Structure）：使用分隔符、编号、标签组织 prompt
4. 示例驱动（Example-driven）：通过 few-shot examples 引导输出

常用 Prompt 技术：
- Chain-of-Thought (CoT)：引导 LLM 逐步推理
  示例：「请一步一步思考这个问题...」
- Few-shot Learning：提供示例来定义输出格式
  示例：「输入：X，输出：Y。现在输入：Z，输出：」
- Role Playing：指定角色来约束行为
  示例：「你是一个资深的 Python 开发工程师...」
- Output Formatting：指定输出格式
  示例：「请以 JSON 格式输出，包含以下字段：...」

Agent 系统中的 Prompt 设计：
1. System Prompt：定义 Agent 的角色、能力、限制
2. Tool Description：描述可用工具的功能和参数
3. Planning Prompt：引导任务分解和规划
4. Reflection Prompt：引导自我评估和修正

Prompt 优化的迭代流程：
1. 基线测试：用初始 prompt 测试效果
2. 错误分析：收集失败案例，分析原因
3. 针对性优化：根据错误模式修改 prompt
4. A/B 测试：对比新旧 prompt 的效果
5. 持续迭代：重复以上过程

常见陷阱：
- Prompt 过长导致关键信息被忽略
- 缺少输出格式约束导致输出不稳定
- 没有处理边界情况（空输入、超长输入）
- 过度依赖 few-shot 导致泛化能力差
""",
        },
        {
            "title": "LangGraph 多智能体编排实战",
            "content": """LangGraph 是 LangChain 团队开发的图状态机框架，专为复杂 Agent 工作流设计。

核心概念：
- StateGraph：定义整个工作流的状态结构和转换规则
- Node：图中的处理节点，每个节点执行一个具体操作
- Edge：节点间的连接，支持普通边和条件边
- State：贯穿整个执行过程的共享状态对象
- Checkpoint：状态快照，支持断点恢复和时间旅行

三种多智能体编排模式：

1. Supervisor 模式
   架构：一个 Supervisor Agent 负责任务路由，多个专业子 Agent 负责执行
   优点：职责清晰，易于扩展新 Agent
   缺点：Supervisor 可能成为瓶颈
   适用：任务类型明确，需要专业分工的场景
   实现要点：
   - Supervisor 通过 LLM 判断用户意图
   - 使用条件边路由到对应子 Agent
   - 子 Agent 完成后返回 Supervisor 决定下一步

2. Swarm 模式
   架构：Agent 之间直接通信和任务传递
   优点：灵活，无中心瓶颈
   缺点：控制流复杂，调试困难
   适用：Agent 之间需要频繁交互的场景

3. Hierarchical 模式
   架构：多层 Agent 组织，Manager 管理 Team Lead，Team Lead 管理 Worker
   优点：适合复杂大规模任务
   缺点：层数多导致延迟增加
   适用：企业级复杂工作流

为什么本项目选择 Supervisor 模式？
1. 面试高频考点：Supervisor 是最常见的多智能体架构
2. 代码量适中：可以在简历项目中完整实现
3. 可扩展性好：新增子 Agent 只需添加节点和边
4. 演示效果好：路由决策过程可以可视化展示

LangGraph 状态设计：
```python
class AgentState(TypedDict):
    messages: list[BaseMessage]      # 对话消息历史
    next_agent: str                   # 下一个要调用的 Agent
    current_agent: str                # 当前执行的 Agent
    tool_results: list[dict]          # 工具调用结果
    context: str                      # RAG 检索上下文
```
""",
        },
        {
            "title": "Function Calling 与工具设计",
            "content": """Function Calling 是 LLM 与外部工具交互的核心机制。

工作原理：
1. 工具定义：开发者定义工具的名称、描述、参数 schema（JSON Schema 格式）
2. 工具选择：LLM 根据用户请求和工具描述，决定是否调用工具
3. 参数生成：LLM 生成符合 schema 的结构化参数
4. 工具执行：系统实际调用工具函数
5. 结果整合：将工具返回值注入对话上下文，LLM 生成最终回答

工具设计最佳实践：
1. 描述要精确：
   差：「搜索文档」
   好：「从知识库中搜索与查询相关的文档片段，返回 Top-K 结果及其相似度分数」

2. 参数要最小化：
   差：10 个可选参数
   好：2-3 个必要参数 + 1-2 个可选参数

3. 错误处理要完善：
   - 工具执行失败时返回友好错误信息
   - 不要让工具异常中断 Agent 流程
   - 记录工具调用日志便于调试

4. 返回格式要结构化：
   差：返回原始 HTML
   好：返回 JSON 格式的结构化数据

安全考虑：
- 代码执行工具必须有沙箱隔离
- API 调用工具需要限流和超时控制
- 文件操作工具需要路径校验，防止目录遍历
- 数据库查询工具需要参数化查询，防止注入

本项目的工具清单：
1. document_search：从 Milvus 检索文档，返回带来源的片段
2. web_search：DuckDuckGo 搜索，返回摘要和链接
3. code_executor：受限 Python 执行，禁止危险操作
4. calculator：数学表达式计算，使用 eval 的安全替代
""",
        },
        {
            "title": "Redis 在 AI 应用中的最佳实践",
            "content": """Redis 在 AI Agent 应用中主要承担对话记忆管理和缓存功能。

对话记忆存储设计：
1. 数据结构选择：
   - List：适合存储消息序列，支持范围查询
   - Hash：适合存储会话元数据
   - String（JSON）：简单场景下直接存储序列化数据
   推荐：使用 List 存储消息，Hash 存储会话元数据

2. Key 设计：
   - 会话消息：chat:{session_id}:messages
   - 会话元数据：chat:{session_id}:meta
   - 用户会话列表：user:{user_id}:sessions

3. 过期策略：
   - 设置 TTL 自动清理过期会话（默认 7 天）
   - 每次访问时刷新 TTL
   - 支持手动清理

上下文窗口管理：
LLM 有 token 限制，需要管理发送给模型的上下文大小：
1. Token 计数：使用 tiktoken 计算消息 token 数
2. 截断策略：
   - 保留 System Prompt（不截断）
   - 保留最近 N 轮对话
   - 截断最早的消息
3. 摘要压缩：将旧对话压缩为摘要，节省 token

缓存策略：
1. Embedding 缓存：缓存文档的 embedding 向量，避免重复计算
2. 检索结果缓存：对相同查询缓存检索结果
3. LLM 回答缓存：对完全相同的输入缓存 LLM 输出

本项目的 Redis 使用：
- 会话存储：List 类型，每个元素是一条 JSON 消息
- TTL：7 天自动过期
- 最大消息数：保留最近 20 条消息（10 轮对话）
- 序列化格式：JSON，包含 role、content、timestamp
""",
        },
        {
            "title": "FastAPI 构建 LLM 应用后端",
            "content": """FastAPI 是构建 LLM 应用后端的理想选择，原因如下：

为什么选 FastAPI 而不是 Flask？
1. 异步支持：FastAPI 原生支持 async/await，适合 LLM 流式调用
2. 自动文档：自动生成 OpenAPI/Swagger 文档
3. 类型安全：基于 Pydantic 的请求/响应验证
4. 性能：基于 Starlette 和 uvicorn，性能优于 Flask
5. SSE 支持：通过 sse-starlette 支持 Server-Sent Events 流式响应

LLM 应用的 API 设计：
1. 流式响应（SSE）：
   - POST /api/chat 返回 text/event-stream
   - 每个 token 作为一个 event 发送
   - 最后发送 [DONE] 标记
   - 前端可以实时展示生成过程

2. 错误处理：
   - 统一错误格式：{"error": "message", "code": "ERROR_CODE"}
   - 区分客户端错误（4xx）和服务端错误（5xx）
   - LLM 超时返回 504

3. 中间件：
   - CORS 中间件：允许前端跨域访问
   - 请求日志中间件：记录请求路径、耗时、状态码
   - 异常处理中间件：捕获未处理异常

流式响应实现要点：
```python
async def stream_response(query: str):
    async for chunk in agent.astream(query):
        yield f"data: {json.dumps({'content': chunk})}\n\n"
    yield "data: [DONE]\n\n"
```

部署注意事项：
- 使用 uvicorn 作为 ASGI 服务器
- 生产环境设置 workers 数量（CPU 核心数 * 2 + 1）
- 配置超时时间（LLM 调用可能较慢）
- 添加健康检查端点 /api/health
""",
        },
        {
            "title": "Docker 容器化 AI 应用部署",
            "content": """Docker 容器化是 AI 应用部署的标准方式，确保环境一致性和可移植性。

AI 应用 Docker 化的挑战：
1. 依赖复杂：Python 包、系统库、CUDA 驱动等
2. 镜像大：ML 框架和模型文件导致镜像体积大
3. 启动慢：模型加载需要时间
4. 资源需求：内存和 GPU 需求高

最佳实践：
1. 多阶段构建：
   - 构建阶段：安装依赖、编译
   - 运行阶段：只包含运行时必需文件
   - 减小镜像体积 50%+

2. 依赖缓存：
   - 先复制 requirements.txt，再安装依赖
   - 利用 Docker 层缓存，加速重复构建

3. 健康检查：
   - 定义 HEALTHCHECK 指令
   - 检查关键服务（API、数据库连接）
   - 设置合理的检查间隔和超时

4. 资源限制：
   - 设置内存限制（--memory）
   - 设置 CPU 限制（--cpus）
   - 防止容器占用过多资源

Docker Compose 编排：
```yaml
services:
  app:
    build: ./backend
    ports: ["8000:8000"]
    depends_on: [milvus, redis]
    environment:
      - MILVUS_HOST=milvus
      - REDIS_HOST=redis

  milvus:
    image: milvusdb/milvus:latest
    ports: ["19530:19530"]

  redis:
    image: redis:latest
    ports: ["6379:6379"]
```

本项目的 Docker 架构：
- app：FastAPI 后端 + Streamlit 前端
- milvus：向量数据库（已有容器）
- redis：缓存/记忆存储（已有容器）
- ollama：本地 Embedding 模型（已有容器）
""",
        },
    ]

    docs = []
    for article in articles:
        doc = {
            "source": "ai_knowledge_base",
            "url": "",
            "title": article["title"],
            "content": article["content"],
            "collected_at": datetime.now().isoformat(),
        }
        docs.append(doc)
        print(f"  [OK] {article['title']} ({len(article['content'])} chars)")

    return docs


def extract_text_from_html(html: str) -> str:
    """简单的 HTML 文本提取"""
    import re
    # 移除 script 和 style 标签
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', ' ', html)
    # 清理空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def save_documents(docs: list, filename: str):
    """保存文档到 JSON 文件"""
    filepath = DATA_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(docs)} documents to {filepath}")
    return filepath


def main():
    print("=" * 60)
    print("数据采集开始")
    print("=" * 60)

    # 1. 采集 AI 领域知识文档
    print("\n[1/2] 生成 AI/Agent/RAG 领域知识文档...")
    ai_docs = collect_ai_knowledge_articles()
    save_documents(ai_docs, "ai_knowledge.json")

    # 2. 采集 LangChain 文档
    print("\n[2/2] 采集 LangChain 官方文档...")
    lc_docs = collect_langchain_docs()
    if lc_docs:
        save_documents(lc_docs, "langchain_docs.json")

    # 汇总
    all_docs = ai_docs + lc_docs
    print("\n" + "=" * 60)
    print(f"采集完成！总计 {len(all_docs)} 篇文档")
    print(f"  - AI 知识文档: {len(ai_docs)} 篇")
    print(f"  - LangChain 文档: {len(lc_docs)} 篇")
    total_chars = sum(len(d["content"]) for d in all_docs)
    print(f"  - 总字数: {total_chars:,} 字符")
    print("=" * 60)


if __name__ == "__main__":
    main()
