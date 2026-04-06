"""Context 内部数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.provider.base import ModelMessage


@dataclass
class ContextEntry:
    """Context 内部项。

    该结构表示上下文层持有的语义材料，而不是最终发送给 provider
    的传输消息。``render()`` 会将其转换为 ``ModelMessage``。
    """

    role: str
    text: str
    kind: str = "message"
    name: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    pinned: bool = False
    priority: int = 0

    def copy(self, **changes: Any) -> "ContextEntry":
        data = {
            "role": self.role,
            "text": self.text,
            "kind": self.kind,
            "name": self.name,
            "tool_calls": [dict(tool_call) for tool_call in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "metadata": dict(self.metadata),
            "extra": dict(self.extra),
            "pinned": self.pinned,
            "priority": self.priority,
        }
        data.update(changes)
        return ContextEntry(**data)

    def to_message(self) -> ModelMessage:
        return ModelMessage(
            role=self.role,
            content=self.text,
            name=self.name,
            tool_calls=[dict(tool_call) for tool_call in self.tool_calls],
            tool_call_id=self.tool_call_id,
            extra=dict(self.extra),
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
            name=message.name,
            tool_calls=[dict(tool_call) for tool_call in message.tool_calls],
            tool_call_id=message.tool_call_id,
            metadata=dict(metadata or {}),
            extra=dict(message.extra),
            pinned=pinned,
            priority=priority,
        )