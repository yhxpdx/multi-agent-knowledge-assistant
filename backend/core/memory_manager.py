"""
长期记忆管理器 — 基于 Milvus 向量库的跨会话记忆系统

架构设计（全链路思考）：
1. 为什么用向量库存储记忆？
   - 语义检索能力：用户说"我之前问过排序算法"，关键词匹配找不到
   - 复用现有 Milvus + Ollama bge-m3，零新增依赖
   - 架构统一：文档 RAG 和记忆检索共享同一套向量搜索基础设施

2. 记忆提取策略：LLM 判断 + 结构化输出
   - 不是每轮都提取（噪声太多），而是 LLM 判断是否值得保存
   - 结构化 JSON 输出可控：should_save / memory / category
   - 只增加一次额外 LLM 调用，开销可控

3. 记忆去重：新记忆与已有记忆 cosine > 0.95 时不重复写入

4. Collection 独立于 documents：
   - 生命周期不同（记忆有 TTL，文档是永久的）
   - 元数据不同（记忆有 category，文档有 doc_id）
   - 可独立管理
"""

import json
import time
import uuid
import logging
from typing import Optional
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

from backend.core.config import get_settings
from backend.core.embedding import get_embedding_service
from backend.core.llm_client import get_llm_client

logger = logging.getLogger(__name__)

VALID_CATEGORIES = ["preference", "fact", "conclusion", "instruction"]

MEMORY_EXTRACT_PROMPT = """你是一个记忆提取器。分析用户和助手的对话，判断是否有值得长期保存的信息。

值得保存的信息类型：
- preference: 用户的偏好（如"我喜欢 Python"、"请用中文回答"）
- fact: 重要事实（如"项目使用 Milvus 做向量检索"）
- conclusion: 对话中产生的结论（如"决定用 HNSW 索引而不是 IVF_FLAT"）
- instruction: 用户给出的指令或约束（如"代码不要加注释"）

不值得保存的信息：
- 普通闲聊（"你好"、"今天天气不错"）
- 一次性的问答（"1+1 等于几"）
- 过于模糊或无意义的内容

请只返回一个 JSON，不要返回其他内容：
{"should_save": true/false, "memory": "提取的记忆内容（简洁，不超过100字）", "category": "preference/fact/conclusion/instruction"}

如果不需要保存，返回：{"should_save": false, "memory": "", "category": ""}"""


class MemoryManager:
    def __init__(self):
        self.settings = get_settings()
        self.collection = self.settings.MEMORY_COLLECTION
        self.client = MilvusClient(
            uri=f"http://{self.settings.MILVUS_HOST}:{self.settings.MILVUS_PORT}"
        )
        self.embedding_service = get_embedding_service()
        self._ensure_collection()

    def _ensure_collection(self):
        """确保 memory_store collection 存在"""
        if self.client.has_collection(self.collection):
            return

        schema = CollectionSchema(fields=[
            FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema("content", DataType.VARCHAR, max_length=512),
            FieldSchema("category", DataType.VARCHAR, max_length=32),
            FieldSchema("session_id", DataType.VARCHAR, max_length=16),
            FieldSchema("created_at", DataType.FLOAT),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=self.settings.EMBEDDING_DIM),
        ])

        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256},
        )
        self.client.create_index(self.collection, index_params)
        logger.info(f"Created collection '{self.collection}' with HNSW index")

    def add_memory(self, content: str, category: str, session_id: str) -> bool:
        """存储一条记忆，含相似度去重（>0.95 不写入）

        Returns: True if memory was added, False if duplicate or error
        """
        if category not in VALID_CATEGORIES:
            category = "fact"

        embedding = self.embedding_service.embed_query(content)
        if not embedding:
            logger.warning(f"Failed to generate embedding for memory: {content[:50]}")
            return False

        # 去重检查：与已有记忆的余弦相似度 > 0.95 时不写入
        existing = self.client.search(
            collection_name=self.collection,
            data=[embedding],
            limit=1,
            output_fields=["id"],
        )
        if existing and existing[0] and existing[0][0]["distance"] > 0.95:
            logger.info(f"Memory already exists (similarity={existing[0][0]['distance']:.2f}), skipping")
            return False

        memory_id = str(uuid.uuid4())[:16]
        self.client.insert(
            collection_name=self.collection,
            data=[{
                "id": memory_id,
                "content": content,
                "category": category,
                "session_id": session_id,
                "created_at": float(time.time()),
                "embedding": embedding,
            }],
        )
        logger.info(f"Added memory: [{category}] {content[:50]}...")
        return True

    def search_memories(self, query: str, top_k: int = 3) -> list[dict]:
        """根据查询语义检索相关记忆

        Returns: list of {"content": str, "category": str, "score": float}
        """
        embedding = self.embedding_service.embed_query(query)
        if not embedding:
            return []

        try:
            results = self.client.search(
                collection_name=self.collection,
                data=[embedding],
                limit=top_k,
                output_fields=["content", "category", "created_at"],
            )
        except Exception as e:
            logger.warning(f"Memory search error: {e}")
            return []

        memories = []
        if results and results[0]:
            for hit in results[0]:
                memories.append({
                    "content": hit["entity"]["content"],
                    "category": hit["entity"]["category"],
                    "score": hit["distance"],
                })
        return memories

    def format_memories(self, memories: list[dict]) -> str:
        """格式化记忆为可注入 prompt 的字符串"""
        if not memories:
            return ""
        lines = [f"[{m['category']}] {m['content']}" for m in memories]
        return "以下是与你相关的记忆：\n" + "\n".join(lines)

    def delete_memory(self, memory_id: str) -> bool:
        """删除指定记忆。返回 True 表示成功，False 表示不存在"""
        try:
            self.client.delete(
                collection_name=self.collection,
                ids=[memory_id],
            )
            return True
        except Exception as e:
            logger.warning(f"Delete memory error: {e}")
            return False

    def list_memories(self, category: Optional[str] = None) -> list[dict]:
        """列出所有记忆，支持按 category 筛选"""
        filter_expr = f'category == "{category}"' if category else None

        try:
            # 使用 query 获取所有记录
            results = self.client.query(
                collection_name=self.collection,
                filter=filter_expr,
                output_fields=["id", "content", "category", "session_id", "created_at"],
            )
            return results
        except Exception as e:
            logger.warning(f"List memories error: {e}")
            return []

    def extract_memory(self, user_msg: str, assistant_msg: str, session_id: str) -> bool:
        """调用 LLM 判断是否提取记忆并写入

        Returns: True if memory was extracted and saved
        """
        llm = get_llm_client()

        conversation = f"用户: {user_msg}\n助手: {assistant_msg}"

        try:
            response = llm.chat([
                {"role": "system", "content": MEMORY_EXTRACT_PROMPT},
                {"role": "user", "content": conversation},
            ], temperature=0, max_tokens=200)

            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            decision = json.loads(response)
            should_save = decision.get("should_save", False)
            memory_content = decision.get("memory", "").strip()
            category = decision.get("category", "fact")

            if not should_save or not memory_content:
                return False

            return self.add_memory(memory_content, category, session_id)

        except json.JSONDecodeError as e:
            logger.warning(f"Memory extraction JSON parse error: {e}")
            return False
        except Exception as e:
            logger.warning(f"Memory extraction error: {e}")
            return False


# 全局单例
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
