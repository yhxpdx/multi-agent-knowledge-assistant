"""
LLM 客户端 — 火山引擎 Doubao 模型

设计决策（全链路思考）：
1. 为什么用 OpenAI 兼容接口？
   - 火山引擎 coding plan 提供 OpenAI 兼容的 API
   - 可以直接使用 openai 库，无需学习新的 SDK
   - 便于后续切换到其他 OpenAI 兼容的模型

2. 为什么封装统一接口？
   - 统一错误处理和重试
   - 支持流式和非流式调用
   - 便于添加日志和监控
"""

from openai import OpenAI
from typing import Optional, Generator
from backend.core.config import get_settings


class LLMClient:
    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(
            api_key=self.settings.ARK_API_KEY,
            base_url=self.settings.ARK_BASE_URL,
        )
        self.model = self.settings.ARK_MODEL_ID

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """非流式对话"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM 调用失败: {e}"

    def chat_stream(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048) -> Generator[str, None, None]:
        """流式对话"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"LLM 调用失败: {e}"

    def health_check(self) -> bool:
        """检查 LLM 服务是否可用"""
        try:
            response = self.chat([{"role": "user", "content": "hi"}], max_tokens=5)
            return "LLM 调用失败" not in response
        except:
            return False


# 全局单例
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
