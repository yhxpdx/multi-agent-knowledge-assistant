"""
文档解析器单元测试

测试覆盖：
1. TXT 文件解析
2. Markdown 文件解析
3. 文本分块逻辑
4. 元数据生成
5. 不存在文件处理
6. 不支持格式处理
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from backend.core.document_parser import DocumentParser, ParsedDocument, get_document_parser


class TestDocumentParser:
    """文档解析器测试"""

    @pytest.fixture
    def parser(self):
        """创建解析器实例，使用 mock settings"""
        with patch("backend.core.document_parser.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                CHUNK_SIZE=500,
                CHUNK_OVERLAP=50,
            )
            return DocumentParser()

    @pytest.fixture
    def txt_file(self, tmp_path):
        """创建临时 TXT 文件"""
        content = "这是第一段测试文本。\n\n这是第二段测试文本，内容更长一些。\n\n这是第三段。"
        file_path = tmp_path / "test.txt"
        file_path.write_text(content, encoding="utf-8")
        return str(file_path)

    @pytest.fixture
    def md_file(self, tmp_path):
        """创建临时 Markdown 文件"""
        content = """# 测试标题

## 第一节

这是第一节的内容，包含一些技术描述。

## 第二节

这是第二节的内容，讨论 RAG 技术。
"""
        file_path = tmp_path / "test.md"
        file_path.write_text(content, encoding="utf-8")
        return str(file_path)

    @pytest.fixture
    def long_txt_file(self, tmp_path):
        """创建较长的 TXT 文件（用于测试分块）"""
        content = "这是一个测试段落。" * 100
        file_path = tmp_path / "long_test.txt"
        file_path.write_text(content, encoding="utf-8")
        return str(file_path)

    def test_parse_txt(self, parser, txt_file):
        """测试 TXT 文件解析"""
        result = parser.parse(txt_file, "test_doc_001")

        assert result is not None
        assert isinstance(result, ParsedDocument)
        assert result.doc_id == "test_doc_001"
        assert result.filename == "test.txt"
        assert result.file_type == ".txt"
        assert result.total_chars > 0
        assert len(result.chunks) > 0

    def test_parse_markdown(self, parser, md_file):
        """测试 Markdown 文件解析"""
        result = parser.parse(md_file, "test_doc_002")

        assert result is not None
        assert result.file_type == ".md"
        assert "测试标题" in result.chunks[0]

    def test_parse_nonexistent_file(self, parser):
        """测试不存在的文件"""
        result = parser.parse("/nonexistent/file.txt", "test_doc_003")
        assert result is None

    def test_parse_unsupported_format(self, parser, tmp_path):
        """测试不支持的文件格式"""
        file_path = tmp_path / "test.xyz"
        file_path.write_text("some content", encoding="utf-8")
        result = parser.parse(str(file_path), "test_doc_004")
        assert result is None

    def test_chunking(self, parser, long_txt_file):
        """测试文本分块"""
        result = parser.parse(long_txt_file, "test_doc_005")

        assert result is not None
        assert len(result.chunks) > 1  # 长文本应该被分成多块

        # 每个 chunk 不应超过 chunk_size (允许一些误差)
        for chunk in result.chunks:
            assert len(chunk) <= 600  # chunk_size=500 + overlap 容差

    def test_metadata_generation(self, parser, txt_file):
        """测试元数据生成"""
        result = parser.parse(txt_file, "test_doc_006")

        assert result is not None
        assert len(result.metadatas) == len(result.chunks)

        for i, meta in enumerate(result.metadatas):
            assert meta["source"] == "test.txt"
            assert meta["chunk_index"] == i
            assert meta["total_chunks"] == len(result.chunks)

    def test_empty_file(self, parser, tmp_path):
        """测试空文件"""
        file_path = tmp_path / "empty.txt"
        file_path.write_text("", encoding="utf-8")
        result = parser.parse(str(file_path), "test_doc_007")

        # 空文件可能返回 None 或空 chunks
        if result is not None:
            assert len(result.chunks) == 0 or result.total_chars == 0

    def test_chinese_content(self, parser, tmp_path):
        """测试中文内容解析"""
        content = "人工智能（AI）是计算机科学的一个分支。它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
        file_path = tmp_path / "chinese.txt"
        file_path.write_text(content, encoding="utf-8")

        result = parser.parse(str(file_path), "test_doc_008")
        assert result is not None
        assert "人工智能" in result.chunks[0]

    def test_get_document_parser_singleton(self):
        """测试单例模式"""
        import backend.core.document_parser as mod
        mod._parser = None  # 重置单例

        with patch("backend.core.document_parser.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(CHUNK_SIZE=500, CHUNK_OVERLAP=50)
            p1 = get_document_parser()
            p2 = get_document_parser()
            assert p1 is p2
