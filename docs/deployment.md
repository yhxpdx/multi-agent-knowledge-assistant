# 部署指南

## 环境要求

- Python 3.11+
- Docker & Docker Compose
- 至少 4GB 内存（Milvus + Ollama 需要）

## 方式一：本地开发部署

### 1. 启动依赖服务

```bash
# Milvus 向量数据库
docker run -d --name milvus \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest

# Redis
docker run -d --name redis \
  -p 6379:6379 \
  redis:latest

# Ollama (本地 Embedding)
docker run -d --name ollama \
  -p 11434:11434 \
  ollama/ollama
```

### 2. 拉取 Embedding 模型

```bash
docker exec ollama ollama pull bge-m3
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

### 5. 导入知识库数据

```bash
python data/collect_data.py  # 采集数据
python data/ingest_data.py   # 导入 Milvus
```

### 6. 启动服务

```bash
# 启动后端
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 启动前端（另一个终端）
streamlit run frontend/app.py
```

### 7. 访问

- 前端：http://localhost:8501
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/api/health

## 方式二：Docker Compose 部署

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env
```

### 2. 一键启动

```bash
docker-compose up -d
```

### 3. 导入数据

```bash
docker exec backend python data/collect_data.py
docker exec backend python data/ingest_data.py
```

### 4. 访问

- 前端：http://localhost:8501
- 后端：http://localhost:8000

## 常见问题

### Milvus 连接失败
检查 Milvus 是否启动：`docker ps | grep milvus`

### Ollama 模型下载慢
使用代理或手动下载模型文件放到 Ollama 数据卷。

### 端口被占用
修改 docker-compose.yml 中的端口映射。
