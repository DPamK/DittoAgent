"""异步 LLM Provider 基类

继承自 BaseProvider，添加异步调用语义：
  - chat(messages, stream=False)  → stream=False 返回 LLMResponse，stream=True 返回 AsyncIterator[str]

具体的异步 Provider 应继承此类并实现 _call_api / _call_api_stream。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, AsyncIterator

from .base import BaseProvider, LLMResponse, ModelMessage, ModelRequest, ProviderConfig


class AsyncBaseProvider(BaseProvider):
    """异步 LLM Provider 抽象基类。

    子类职责：
    - 实现 ``_build_request`` / ``_parse_response`` / ``_parse_chunk``（来自 BaseProvider）
    - 实现 ``_call_api``        — 发起异步非流式请求，返回原始响应
    - 实现 ``_call_api_stream`` — 发起异步流式请求，返回原始 chunk 异步迭代器

    公共接口：
    - ``chat(messages, stream=False)`` — 统一入口，按 stream 参数决定返回类型
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # 内部抽象接口
    # ------------------------------------------------------------------

    @abstractmethod
    async def _call_api(self, request: dict[str, Any]) -> Any:
        """向 LLM 发起异步非流式请求。

        Args:
            request: 由 ``_build_request`` 生成的请求字典。

        Returns:
            底层 SDK 的原始响应对象，将被传入 ``_parse_response``。
        """

    @abstractmethod
    async def _call_api_stream(self, request: dict[str, Any]) -> AsyncIterator[Any]:
        """向 LLM 发起异步流式请求。

        Args:
            request: 由 ``_build_request`` 生成的请求字典。

        Yields:
            底层 SDK 的流式 chunk 对象，将被逐一传入 ``_parse_chunk``。
        """

    # ------------------------------------------------------------------
    # 公共调用接口
    # ------------------------------------------------------------------

    async def chat(
        self,
        request_or_messages: ModelRequest | list[ModelMessage],
        stream: bool = False,
        **kwargs: Any,
    ) -> "LLMResponse | AsyncIterator[str]":
        """统一调用入口。

        Args:
            request_or_messages: 结构化请求或对话消息列表。
            stream:   False → 返回完整 LLMResponse；True → 返回逐 token 的 AsyncIterator[str]。
            **kwargs: 运行时覆盖参数（优先级高于 config）。

        Returns:
            stream=False 时返回 LLMResponse；
            stream=True  时返回 AsyncIterator[str]，可直接 async for 迭代。
        """
        request = self._ensure_request(request_or_messages)
        if stream:
            return self._iter_stream(request, **kwargs)
        payload = self._build_request(request, **kwargs)
        raw = await self._call_api(payload)
        return self._parse_response(raw)

    async def _iter_stream(
        self, request: ModelRequest, **kwargs: Any
    ) -> AsyncIterator[str]:
        """内部辅助：构造异步流式迭代器。"""
        payload = self._build_request(request, stream=True, **kwargs)
        async for chunk in await self._call_api_stream(payload):
            text = self._parse_chunk(chunk)
            if text:
                yield text
