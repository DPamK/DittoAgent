"""LLM Provider 模块 — 抽象基类定义

层级结构：
    BaseProvider          ← 通用抽象基类（数据结构 + 公共接口）
    ├── SyncBaseProvider  ← 同步基类（在 sync_base.py 中）
    └── AsyncBaseProvider ← 异步基类（在 async_base.py 中）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class ModelMessage:
    """单条对话消息"""
    role: str   # "system" | "user" | "assistant"
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def copy(self, **changes: Any) -> "ModelMessage":
        data = {
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "tool_calls": [dict(tool_call) for tool_call in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "extra": dict(self.extra),
        }
        data.update(changes)
        return ModelMessage(**data)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
        }

        if self.content is not None:
            payload["content"] = self.content
        elif self.tool_calls:
            payload["content"] = ""

        if self.name:
            payload["name"] = self.name

        if self.tool_calls:
            payload["tool_calls"] = [dict(tool_call) for tool_call in self.tool_calls]

        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id

        for key, value in self.extra.items():
            if value is not None:
                payload[key] = value

        return payload


@dataclass
class ModelRequest:
    """一次模型调用的完整请求结构。"""

    messages: list[ModelMessage]
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    def copy(self, **changes: Any) -> "ModelRequest":
        data = {
            "messages": [message.copy() for message in self.messages],
            "tools": [dict(tool_schema) for tool_schema in self.tools],
            "tool_choice": self.tool_choice,
            "extra": dict(self.extra),
        }
        data.update(changes)
        return ModelRequest(**data)


@dataclass
class ProviderConfig:
    """Provider 公共配置项"""
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    timeout: float = 30.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """LLM 调用的统一响应结构"""
    content: str
    model: str
    usage: dict[str, int]   # prompt_tokens / completion_tokens / total_tokens
    message: Optional[ModelMessage] = None
    raw: Any = None         # 保留底层 SDK 原始响应，便于调试


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class BaseProvider(ABC):
    """所有 LLM Provider 的公共抽象基类。

    职责：
    - 持有配置 (ProviderConfig)
    - 声明请求构建 / 响应解析 / 流式 chunk 解析三个内部抽象接口
    - 不绑定任何同步或异步调用语义（由子类决定）
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # 公共属性
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        return self.config.model

    # ------------------------------------------------------------------
    # 内部抽象接口（供子类实现）
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_request(self, request: ModelRequest, **kwargs: Any) -> dict[str, Any]:
        """将统一的 ModelRequest 转换为底层 SDK 所需的请求参数字典。

        Args:
            request: 结构化模型请求。
            **kwargs: 运行时覆盖参数（如临时调整 temperature 等）。

        Returns:
            可直接传给 SDK 的请求字典。
        """

    def _ensure_request(self, request_or_messages: ModelRequest | list[ModelMessage]) -> ModelRequest:
        if isinstance(request_or_messages, ModelRequest):
            return request_or_messages
        return ModelRequest(messages=[message.copy() for message in request_or_messages])

    @abstractmethod
    def _parse_response(self, raw_response: Any) -> LLMResponse:
        """将 SDK 原始响应解析为标准 LLMResponse。

        Args:
            raw_response: 底层 SDK 返回的原始对象。

        Returns:
            标准化的 LLMResponse。
        """

    @abstractmethod
    def _parse_chunk(self, chunk: Any) -> str:
        """将流式响应的单个 chunk 解析为文本片段。

        Args:
            chunk: 底层 SDK 的流式 chunk 对象。

        Returns:
            该 chunk 对应的文本内容字符串。
        """
