"""OpenAI Provider 实现（同步 + 异步同文件）。

本文件提供两个类：
- OpenAIProvider: 同步实现，继承 SyncBaseProvider
- AsyncOpenAIProvider: 异步实现，继承 AsyncBaseProvider

环境变量：
- OPENAI_API_KEY
- OPENAI_BASE_URL（可选，默认 https://api.openai.com/v1）
"""

from __future__ import annotations

import os
import importlib
from typing import Any, AsyncIterator, Iterator

from .async_base import AsyncBaseProvider
from .base import LLMResponse, ModelMessage, ModelRequest, ProviderConfig
from .sync_base import SyncBaseProvider


class _OpenAIMixin:
    """OpenAI 公共实现片段，供同步/异步 Provider 复用。"""

    def _build_request(self, request: ModelRequest, **kwargs: Any) -> dict[str, Any]:
        req: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in request.messages],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            **self.config.extra_kwargs,
            **request.extra,
            **kwargs,
        }

        if request.tools:
            req["tools"] = [dict(tool_schema) for tool_schema in request.tools]

        if request.tool_choice is not None and "tool_choice" not in req:
            req["tool_choice"] = request.tool_choice

        if self.config.max_tokens is not None and "max_tokens" not in req:
            req["max_tokens"] = self.config.max_tokens

        return req

    def _parse_response(self, raw_response: Any) -> LLMResponse:
        content = ""
        parsed_message: ModelMessage | None = None
        if getattr(raw_response, "choices", None):
            first_choice = raw_response.choices[0]
            message = getattr(first_choice, "message", None)
            if message is not None:
                content = getattr(message, "content", "") or ""
                tool_calls_raw = getattr(message, "tool_calls", None) or []
                normalized_content = getattr(message, "content", None)
                if normalized_content is None and tool_calls_raw:
                    normalized_content = ""
                parsed_message = ModelMessage(
                    role=getattr(message, "role", "assistant") or "assistant",
                    content=normalized_content,
                    tool_calls=[tc.model_dump() if hasattr(tc, "model_dump") else dict(tc) for tc in tool_calls_raw],
                )

        usage_obj = getattr(raw_response, "usage", None)
        usage = {
            "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
        }

        model = getattr(raw_response, "model", self.config.model)
        return LLMResponse(content=content, model=model, usage=usage, message=parsed_message, raw=raw_response)

    def _parse_chunk(self, chunk: Any) -> str:
        choices = getattr(chunk, "choices", None)
        if not choices:
            return ""
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            return ""
        return getattr(delta, "content", "") or ""


class OpenAIProvider(_OpenAIMixin, SyncBaseProvider):
    """OpenAI 同步 Provider。"""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("缺少环境变量 OPENAI_API_KEY")

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        try:
            openai_module = importlib.import_module("openai")
            openai_client_cls = getattr(openai_module, "OpenAI")
        except Exception as exc:
            raise ImportError("未安装 openai 包，请先执行: pip install openai") from exc

        self.client = openai_client_cls(
            api_key=api_key,
            base_url=base_url,
            timeout=self.config.timeout,
        )

    def _call_api(self, request: dict[str, Any]) -> Any:
        return self.client.chat.completions.create(**request)

    def _call_api_stream(self, request: dict[str, Any]) -> Iterator[Any]:
        return self.client.chat.completions.create(**request)


class AsyncOpenAIProvider(_OpenAIMixin, AsyncBaseProvider):
    """OpenAI 异步 Provider。"""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("缺少环境变量 OPENAI_API_KEY")

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        try:
            openai_module = importlib.import_module("openai")
            async_openai_client_cls = getattr(openai_module, "AsyncOpenAI")
        except Exception as exc:
            raise ImportError("未安装 openai 包，请先执行: pip install openai") from exc

        self.client = async_openai_client_cls(
            api_key=api_key,
            base_url=base_url,
            timeout=self.config.timeout,
        )

    async def _call_api(self, request: dict[str, Any]) -> Any:
        return await self.client.chat.completions.create(**request)

    async def _call_api_stream(self, request: dict[str, Any]) -> AsyncIterator[Any]:
        return await self.client.chat.completions.create(**request)
