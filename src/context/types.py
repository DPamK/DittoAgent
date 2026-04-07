"""Context 内部数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.provider.base import ModelMessage


@dataclass
class ContextTransport:
    """ContextEntry 关联的 provider 传输态。

    该结构只保存为了和 provider 消息无损往返而保留的字段，不承载
    context 自身的裁剪、检索、记忆等语义信息。
    """

    name: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def copy(self, **changes: Any) -> "ContextTransport":
        data = {
            "name": self.name,
            "tool_calls": [dict(tool_call) for tool_call in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "extra": dict(self.extra),
        }
        data.update(changes)
        return ContextTransport(**data)


@dataclass
class ContextEntry:
    """Context 内部项。

    该结构表示上下文层持有的语义材料，而不是最终发送给 provider
    的传输消息。``render()`` 会将其转换为 ``ModelMessage``。
    """

    role: str
    text: str
    kind: str = "message"
    metadata: dict[str, Any] = field(default_factory=dict)
    transport: ContextTransport = field(default_factory=ContextTransport)
    pinned: bool = False
    priority: int = 0

    def copy(self, **changes: Any) -> "ContextEntry":
        data = {
            "role": self.role,
            "text": self.text,
            "kind": self.kind,
            "metadata": dict(self.metadata),
            "transport": self.transport.copy(),
            "pinned": self.pinned,
            "priority": self.priority,
        }
        data.update(changes)
        return ContextEntry(**data)

    def to_message(self) -> ModelMessage:
        return ModelMessage(
            role=self.role,
            content=self.text,
            name=self.transport.name,
            tool_calls=[dict(tool_call) for tool_call in self.transport.tool_calls],
            tool_call_id=self.transport.tool_call_id,
            extra=dict(self.transport.extra),
        )

    @classmethod
    def from_message(
        cls,
        message: ModelMessage,
        *,
        kind: str = "message",
        metadata: dict[str, Any] | None = None,
        pinned: bool = False,
        priority: int = 0,
    ) -> "ContextEntry":
        return cls(
            role=message.role,
            text=message.content or "",
            kind=kind,
            metadata=dict(metadata or {}),
            transport=ContextTransport(
                name=message.name,
                tool_calls=[dict(tool_call) for tool_call in message.tool_calls],
                tool_call_id=message.tool_call_id,
                extra=dict(message.extra),
            ),
            pinned=pinned,
            priority=priority,
        )