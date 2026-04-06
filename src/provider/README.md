# Provider 模块使用说明

本目录提供了 LLM Provider 的统一抽象，目标是让上层 Agent/Context 层不依赖具体厂商 SDK。

## 目录结构

- `base.py`
    - `ModelMessage`: 统一消息结构
    - `ModelRequest`: 统一请求结构
  - `ProviderConfig`: Provider 默认配置
  - `LLMResponse`: 统一响应结构
  - `BaseProvider`: 通用抽象基类
- `sync_base.py`
  - `SyncBaseProvider`: 同步调用基类，统一入口 `chat(messages, stream=False)`
- `async_base.py`
  - `AsyncBaseProvider`: 异步调用基类，统一入口 `chat(messages, stream=False)`

## 核心概念

### 1) ModelMessage

```python
ModelMessage(role="user", content="你好")
```

- `role`: 推荐使用 `system` / `user` / `assistant`
- `content`: 文本内容
- `name`: 可选消息名，用于兼容具名消息场景
- `extra`: 发送到底层 provider 的额外字段扩展槽

### 2) ProviderConfig

Provider 的默认参数配置，会在 Provider 初始化时注入：

```python
ProviderConfig(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=512,
    top_p=1.0,
    timeout=30.0,
    extra_kwargs={"seed": 42},
)
```

### 3) 统一调用接口 chat

无论同步还是异步，统一使用 `chat(request_or_messages, stream=False)`：

- `stream=False`：返回完整响应 `LLMResponse`
- `stream=True`：返回流式迭代器（同步 `Iterator[str]`，异步 `AsyncIterator[str]`）

推荐传入 `ModelRequest`，也兼容直接传 `list[ModelMessage]`。

## 如何使用（同步）

```python
from src.provider import ModelMessage, ModelRequest, ProviderConfig
from your_provider import MySyncProvider

provider = MySyncProvider(
    ProviderConfig(model="gpt-4o-mini", temperature=0.2)
)

messages = [
    ModelMessage(role="system", content="你是一个助手"),
    ModelMessage(role="user", content="写一句欢迎词"),
]

request = ModelRequest(messages=messages)

# 非流式
resp = provider.chat(request)
print(resp.content)
print(resp.usage)

# 流式
for token in provider.chat(request, stream=True):
    print(token, end="")
```

## 如何使用（异步）

```python
import asyncio

from src.provider import ModelMessage, ModelRequest, ProviderConfig
from your_provider import MyAsyncProvider


async def main() -> None:
    provider = MyAsyncProvider(
        ProviderConfig(model="gpt-4o-mini", temperature=0.2)
    )

    messages = [
        ModelMessage(role="system", content="你是一个助手"),
        ModelMessage(role="user", content="写一句欢迎词"),
    ]

    request = ModelRequest(messages=messages)

    # 非流式
    resp = await provider.chat(request)
    print(resp.content)

    # 流式
    stream_iter = await provider.chat(request, stream=True)
    async for token in stream_iter:
        print(token, end="")


asyncio.run(main())
```

## 如何 DIY 一个 Provider

下面分别给出同步和异步的最小骨架，你只需要实现抽象方法即可。

### DIY 同步 Provider

```python
from typing import Any, Iterator

from src.provider import (
    LLMResponse,
    ModelMessage,
    ProviderConfig,
    SyncBaseProvider,
)


class MySyncProvider(SyncBaseProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        # 在这里初始化你的 SDK Client
        # 例如: self.client = SomeClient(api_key=...)

    def _build_request(self, messages: list[ModelMessage], **kwargs: Any) -> dict[str, Any]:
        # 1) 将通用 ModelMessage 转成 SDK 需要的格式
        sdk_messages = [m.to_dict() for m in messages]

        # 2) 配置默认参数 + 调用时覆盖参数
        req = {
            "model": self.config.model,
            "messages": sdk_messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "timeout": self.config.timeout,
            **self.config.extra_kwargs,
            **kwargs,
        }
        if self.config.max_tokens is not None and "max_tokens" not in req:
            req["max_tokens"] = self.config.max_tokens
        return req

    def _call_api(self, request: dict[str, Any]) -> Any:
        # 发起非流式请求，返回 SDK 原始响应
        # return self.client.chat.completions.create(**request)
        raise NotImplementedError

    def _call_api_stream(self, request: dict[str, Any]) -> Iterator[Any]:
        # 发起流式请求，返回 chunk 迭代器
        # return self.client.chat.completions.create(**request)
        raise NotImplementedError

    def _parse_response(self, raw_response: Any) -> LLMResponse:
        # 从 SDK 原始响应中提取标准字段
        # 以下字段名按你的 SDK 实际结构替换
        content = raw_response.choices[0].message.content
        usage = {
            "prompt_tokens": getattr(raw_response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(raw_response.usage, "completion_tokens", 0),
            "total_tokens": getattr(raw_response.usage, "total_tokens", 0),
        }
        model = getattr(raw_response, "model", self.config.model)
        return LLMResponse(content=content, model=model, usage=usage, raw=raw_response)

    def _parse_chunk(self, chunk: Any) -> str:
        # 从 stream chunk 提取本次增量文本
        # 典型结构可能是: chunk.choices[0].delta.content
        try:
            return chunk.choices[0].delta.content or ""
        except Exception:
            return ""
```

### DIY 异步 Provider

```python
from typing import Any, AsyncIterator

from src.provider import (
    AsyncBaseProvider,
    LLMResponse,
    ModelMessage,
    ProviderConfig,
)


class MyAsyncProvider(AsyncBaseProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        # 初始化异步 SDK Client

    def _build_request(self, messages: list[ModelMessage], **kwargs: Any) -> dict[str, Any]:
        sdk_messages = [m.to_dict() for m in messages]
        req = {
            "model": self.config.model,
            "messages": sdk_messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "timeout": self.config.timeout,
            **self.config.extra_kwargs,
            **kwargs,
        }
        if self.config.max_tokens is not None and "max_tokens" not in req:
            req["max_tokens"] = self.config.max_tokens
        return req

    async def _call_api(self, request: dict[str, Any]) -> Any:
        # return await self.client.chat.completions.create(**request)
        raise NotImplementedError

    async def _call_api_stream(self, request: dict[str, Any]) -> AsyncIterator[Any]:
        # return await self.client.chat.completions.create(**request)
        raise NotImplementedError

    def _parse_response(self, raw_response: Any) -> LLMResponse:
        content = raw_response.choices[0].message.content
        usage = {
            "prompt_tokens": getattr(raw_response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(raw_response.usage, "completion_tokens", 0),
            "total_tokens": getattr(raw_response.usage, "total_tokens", 0),
        }
        model = getattr(raw_response, "model", self.config.model)
        return LLMResponse(content=content, model=model, usage=usage, raw=raw_response)

    def _parse_chunk(self, chunk: Any) -> str:
        try:
            return chunk.choices[0].delta.content or ""
        except Exception:
            return ""
```

## 实现建议

- `_build_request` 里始终做“默认参数 + 运行时覆盖”的合并。
- `_parse_response` 统一返回 `LLMResponse`，上层只处理标准结构。
- `_parse_chunk` 尽量容错，遇到空 chunk 返回空字符串即可。
- SDK 原始响应建议放到 `LLMResponse.raw`，方便排查线上问题。

## 常见坑

- 同步模式流式调用时：`for token in provider.chat(..., stream=True)`
- 异步模式流式调用时：
  - 先 `stream_iter = await provider.chat(..., stream=True)`
  - 再 `async for token in stream_iter`
- 不同厂商 `usage` 字段差异较大，建议在 `_parse_response` 做兜底。
