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
class Message:
    """单条对话消息"""
    role: str   # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


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
    def _build_request(self, messages: list[Message], **kwargs: Any) -> dict[str, Any]:
        """将统一的 Message 列表转换为底层 SDK 所需的请求参数字典。

        Args:
            messages: 对话消息列表。
            **kwargs: 运行时覆盖参数（如临时调整 temperature 等）。

        Returns:
            可直接传给 SDK 的请求字典。
        """

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
