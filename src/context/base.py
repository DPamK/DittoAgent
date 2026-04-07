"""Context 抽象基类与公共协议。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from src.provider.base import ModelMessage, ModelRequest

from .types import ContextEntry


class BaseContext(ABC):
    """所有 Context 实现的公共抽象基类。

    Context 负责管理消息生命周期，并通过 ``render()`` 统一产出
    provider 可消费的 ``ModelRequest``。
    """

    def __init__(
        self,
        items: Iterable[ContextEntry] | None = None,
        messages: Iterable[ModelMessage] | None = None,
    ) -> None:
        if items is not None:
            self.add_items(items)
        if messages is not None:
            self.add_messages(messages)

    @property
    def items(self) -> tuple[ContextEntry, ...]:
        """返回当前上下文项快照，避免暴露内部可变列表。"""
        return tuple(self._clone_items(self._list_items()))

    @property
    def messages(self) -> tuple[ModelMessage, ...]:
        """返回渲染后的消息快照，兼容简单聊天场景。"""
        return tuple(self.render_messages())

    def add_item(self, item: ContextEntry) -> ContextEntry:
        """追加单条上下文项。"""
        normalized = self._normalize_item(item)
        self._before_store_item(normalized)
        self._store_item(normalized)
        self._after_store_item(normalized)
        return normalized.copy()

    def add_items(self, items: Iterable[ContextEntry]) -> list[ContextEntry]:
        """批量追加上下文项。"""
        added_items: list[ContextEntry] = []
        for item in items:
            added_items.append(self.add_item(item))
        return added_items

    def add_message(self, message: ModelMessage) -> ContextEntry:
        """将外部消息转换为 ContextEntry 后入库。"""
        return self.add_item(ContextEntry.from_message(message, kind="message"))

    def add_response_message(self, message: ModelMessage) -> ContextEntry:
        """接收 provider 返回消息并以响应语义回填到 context。"""
        return self.add_item(
            ContextEntry.from_message(
                message,
                kind="provider_response",
                metadata={"source": "provider"},
            )
        )

    def add_messages(self, messages: Iterable[ModelMessage]) -> list[ContextEntry]:
        """批量接收外部消息并转换为 ContextEntry。"""
        added_items: list[ContextEntry] = []
        for message in messages:
            added_items.append(self.add_message(message))
        return added_items

    def add_response_messages(self, messages: Iterable[ModelMessage]) -> list[ContextEntry]:
        """批量接收 provider 返回消息并回填到 context。"""
        added_items: list[ContextEntry] = []
        for message in messages:
            added_items.append(self.add_response_message(message))
        return added_items

    def clear(self) -> None:
        """清空上下文中的所有消息。"""
        self._before_clear()
        self._clear_items()
        self._after_clear()

    def render(self) -> ModelRequest:
        """将当前上下文渲染为 provider 可消费的请求对象。"""
        items = self._clone_items(self._list_items())
        items = self._prepare_items_for_render(items)
        messages = self._render_items(items)
        finalized_messages = self._clone_messages(self._finalize_rendered_messages(messages))
        tools = self._clone_tools(self._finalize_rendered_tools(self._render_tools(items)))
        request = ModelRequest(messages=finalized_messages, tools=tools)
        return self._finalize_rendered_request(request)

    def render_messages(self) -> list[ModelMessage]:
        """返回渲染后的消息列表。"""
        return [message.copy() for message in self.render().messages]

    def render_tools(self) -> list[dict[str, Any]]:
        """返回最终渲染结果中的工具 schema，等价于 render().tools。"""
        return [dict(tool_schema) for tool_schema in self.render().tools]

    def _normalize_item(self, item: ContextEntry) -> ContextEntry:
        """标准化上下文项，供 mixin 在入库前扩展。"""
        return item.copy()

    def _before_store_item(self, item: ContextEntry) -> None:
        """上下文项入库前钩子。"""

    def _after_store_item(self, item: ContextEntry) -> None:
        """上下文项入库后钩子。"""

    def _before_clear(self) -> None:
        """清空前钩子。"""

    def _after_clear(self) -> None:
        """清空后钩子。"""

    def _prepare_items_for_render(self, items: list[ContextEntry]) -> list[ContextEntry]:
        """渲染前增强钩子。"""
        return items

    def _render_items(self, items: list[ContextEntry]) -> list[ModelMessage]:
        """核心渲染钩子，默认将上下文项映射为 provider ModelMessage。"""
        return [item.to_message() for item in items]

    def _render_tools(self, items: list[ContextEntry]) -> list[dict[str, Any]]:
        """渲染工具 schema。"""
        return []

    def _finalize_rendered_messages(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """渲染结果收尾钩子。"""
        return messages

    def _finalize_rendered_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """工具渲染结果收尾钩子。"""
        return tools

    def _finalize_rendered_request(self, request: ModelRequest) -> ModelRequest:
        """请求渲染结果收尾钩子。"""
        return request.copy()

    def _clone_items(self, items: Iterable[ContextEntry]) -> list[ContextEntry]:
        return [item.copy() for item in items]

    def _clone_messages(self, messages: Iterable[ModelMessage]) -> list[ModelMessage]:
        return [message.copy() for message in messages]

    def _clone_tools(self, tools: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [dict(tool_schema) for tool_schema in tools]

    @abstractmethod
    def _store_item(self, item: ContextEntry) -> None:
        """将单条上下文项写入底层存储。"""

    @abstractmethod
    def _list_items(self) -> list[ContextEntry]:
        """返回内部上下文项列表。"""

    @abstractmethod
    def _clear_items(self) -> None:
        """清空底层上下文项存储。"""