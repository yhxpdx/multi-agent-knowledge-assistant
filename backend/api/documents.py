"""
文档管理 API

端点设计：
- POST /api/documents: 上传文档
- GET /api/documents: 列出所有文档
- DELETE /api/documents/{doc_id}: 删除文档
"""

import os
import uuid
import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from backend.core.document_parser import get_document_parser
from backend.core.embedding import get_embedding_service
from backend.core.milvus_client import get_milvus_manager
from backend.core.config import get_settings

router = APIRouter(prefix="/api/documents", tags=["documents"])

# 文档元数据存储（简单文件存储）
UPLOAD_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
META_FILE = UPLOAD_DIR / "metadata.json"


def _load_metadata() -> dict:
    if META_FILE.exists():
        with open(META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_metadata(meta: dict):
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    size_bytes: int
    total_chars: int
    chunk_count: int


@router.post("", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    """上传文档"""
    settings = get_settings()

    # 验证文件类型
    allowed_types = {".pdf", ".txt", ".md", ".docx"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_types:
        raise HTTPException(400, f"不支持的文件类型: {suffix}，支持: {allowed_types}")

    # 读取文件内容
    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"文件过大，最大 {settings.MAX_FILE_SIZE_MB}MB")

    # 保存文件
    doc_id = str(uuid.uuid4())[:8]
    file_path = UPLOAD_DIR / f"{doc_id}{suffix}"
    with open(file_path, "wb") as f:
        f.write(content)

    # 解析文档
    parser = get_document_parser()
    parsed = parser.parse(str(file_path), doc_id)
    if not parsed:
        os.remove(file_path)
        raise HTTPException(500, "文档解析失败")

    # 生成 embedding
    embedding_service = get_embedding_service()
    embeddings = embedding_service.embed_documents(parsed.chunks)
    if not all(embeddings):
        os.remove(file_path)
        raise HTTPException(500, "Embedding 生成失败")

    # 存储到 Milvus
    milvus = get_milvus_manager()
    milvus.create_collection()
    milvus.insert(doc_id, parsed.chunks, embeddings, parsed.metadatas)

    # 保存元数据
    meta = _load_metadata()
    meta[doc_id] = {
        "filename": file.filename,
        "file_type": suffix,
        "size_bytes": len(content),
        "total_chars": parsed.total_chars,
        "chunk_count": len(parsed.chunks),
    }
    _save_metadata(meta)

    return DocumentInfo(
        doc_id=doc_id,
        filename=file.filename,
        file_type=suffix,
        size_bytes=len(content),
        total_chars=parsed.total_chars,
        chunk_count=len(parsed.chunks),
    )


@router.get("")
async def list_documents():
    """列出所有文档"""
    meta = _load_metadata()
    return [
        {"doc_id": did, **info}
        for did, info in meta.items()
    ]


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档"""
    meta = _load_metadata()
    if doc_id not in meta:
        raise HTTPException(404, "文档不存在")

    # 从 Milvus 删除
    milvus = get_milvus_manager()
    milvus.delete_by_doc(doc_id)

    # 删除本地文件
    info = meta[doc_id]
    file_path = UPLOAD_DIR / f"{doc_id}{info['file_type']}"
    if file_path.exists():
        os.remove(file_path)

    # 删除元数据
    del meta[doc_id]
    _save_metadata(meta)

    return {"message": f"文档 {doc_id} 已删除"}
