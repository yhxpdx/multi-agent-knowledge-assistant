"""
联网搜索工具

设计决策：
1. 为什么选 DuckDuckGo？
   - 免费，无需 API Key
   - 不需要注册，开箱即用
   - 支持中文搜索
   - 适合项目演示，无成本

2. 为什么限制结果数量？
   - 避免占用过多 context
   - 3 条结果足以提供参考
"""

from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """搜索互联网获取实时信息。当需要最新资讯、事实核查或知识库未覆盖的信息时使用。

    Args:
        query: 搜索查询
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "未找到相关搜索结果。"

            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                formatted.append(f"[{i}] {title}\n{body}\n链接: {href}")

            return "\n\n".join(formatted)
    except ImportError:
        return "DuckDuckGo 搜索库未安装，请运行: pip install duckduckgo-search"
    except Exception as e:
        return f"搜索失败: {e}"
