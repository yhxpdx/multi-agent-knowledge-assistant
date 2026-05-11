"""
FastAPI 应用主入口

启动方式：
  cd multi-agent-knowledge-assistant
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.chat import router as chat_router
from backend.api.documents import router as docs_router
from backend.api.sessions import router as sessions_router
from backend.api.health import router as health_router
from backend.api.memories import router as memories_router

app = FastAPI(
    title="Multi-Agent Knowledge Assistant",
    description="多智能体知识助手 API",
    version="1.0.0",
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router)
app.include_router(docs_router)
app.include_router(sessions_router)
app.include_router(health_router)
app.include_router(memories_router)


@app.get("/")
async def root():
    return {
        "name": "Multi-Agent Knowledge Assistant",
        "version": "1.0.0",
        "docs": "/docs",
    }
