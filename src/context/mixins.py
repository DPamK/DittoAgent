"""Context mixin 扩展骨架。"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from src.provider.base import ModelMessage, ModelRequest

from .tools import BaseTool, ToolRegistry, ToolResult, ensure_tool
from .types import ContextEntry


class MessageTransformMixin:
    """消息入库前变换扩展点。"""

    def _normalize_item(self, item: ContextEntry) -> ContextEntry:
        item = super()._normalize_item(item)
        return self.transform_message(item)

    def transform_message(self, item: ContextEntry) -> ContextEntry:
        """自定义单条上下文项的标准化行为。"""
        return item


class RenderTransformMixin:
    """渲染阶段变换扩展点。"""

    def _prepare_items_for_render(self, items: list[ContextEntry]) -> list[ContextEntry]:
        items = super()._prepare_items_for_render(items)
        return self.transform_render_items(items)

    def transform_render_items(self, items: list[ContextEntry]) -> list[ContextEntry]:
        """自定义渲染前的上下文项增强逻辑。"""
        return items


class MessageMetadataMixin(MessageTransformMixin):
    """通过 metadata 扩展消息的基础 mixin。"""

    def transform_message(self, item: ContextEntry) -> ContextEntry:
        item = super().transform_message(item)
        metadata = dict(item.metadata)
        metadata.update(self.build_message_metadata(item))
        return item.copy(metadata=metadata)

    def build_message_metadata(self, item: ContextEntry) -> Mapping[str, object]:
        """返回需要附加到上下文项上的 metadata。"""
        return {}


class ToolsContextMixin(RenderTransformMixin):
    """Tools 相关渲染增强扩展点。"""

    def __init__(
        self,
        *args: Any,
        tools: Iterable[BaseTool | Callable[..., Any]] | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_render_mode: str = "native",
        **kwargs: Any,
    ) -> None:
        self._tool_registry = tool_registry or ToolRegistry()
        self._tool_render_mode = tool_render_mode
        if tools is not None:
            self.register_tools(tools)
        super().__init__(*args, **kwargs)

    def transform_render_items(self, items: list[ContextEntry]) -> list[ContextEntry]:
        items = super().transform_render_items(items)
        return self.inject_tool_items(items)

    def inject_tool_items(self, items: list[ContextEntry]) -> list[ContextEntry]:
        """返回渲染前需要额外注入的 tools 相关上下文项。"""
        return items

    def build_tool_messages(self) -> list[ModelMessage]:
        """兼容降级模式：将 tools 注入为消息。"""
        return []

    def _render_tools(self, items: list[ContextEntry]) -> list[dict[str, Any]]:
        if self._tool_render_mode != "native":
            return []
        return self.get_tool_schemas()

    def _finalize_rendered_messages(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        messages = super()._finalize_rendered_messages(messages)
        if self._tool_render_mode != "prompt":
            return messages
        tool_messages = [message.copy() for message in self.build_tool_messages()]
        if not tool_messages:
            return messages
        return [*tool_messages, *messages]

    def _finalize_rendered_request(self, request: ModelRequest) -> ModelRequest:
        request = super()._finalize_rendered_request(request)
        if self._tool_render_mode == "native":
            return request.copy(tools=self.get_tool_schemas())
        return request

    def register_tool(self, tool: BaseTool | Callable[..., Any], *, replace: bool = False) -> BaseTool:
        return self._tool_registry.register(ensure_tool(tool), replace=replace)

    def register_tools(
        self,
        tools: Iterable[BaseTool | Callable[..., Any]],
        *,
        replace: bool = False,
    ) -> list[BaseTool]:
        registered: list[BaseTool] = []
        for tool in tools:
            registered.append(self.register_tool(tool, replace=replace))
        return registered

    def get_tool(self, name: str) -> BaseTool:
        return self._tool_registry.get(name)

    def list_tools(self) -> list[BaseTool]:
        return self._tool_registry.list()

    def render_tools(self) -> list[dict[str, Any]]:
        return self._tool_registry.schemas()

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return self.render_tools()

    def invoke_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        payload = dict(arguments or {})
        payload.update(kwargs)
        return self._tool_registry.invoke(name, **payload)


class SkillsContextMixin(RenderTransformMixin):
    """Skills 相关渲染增强扩展点。"""

    def transform_render_items(self, items: list[ContextEntry]) -> list[ContextEntry]:
        items = super().transform_render_items(items)
        return self.inject_skill_items(items)

    def _finalize_rendered_messages(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        messages = super()._finalize_rendered_messages(messages)
        skill_messages = [message.copy() for message in self.build_skill_messages()]
        if not skill_messages:
            return messages
        return [*skill_messages, *messages]

    def inject_skill_items(self, items: list[ContextEntry]) -> list[ContextEntry]:
        """返回渲染前需要额外注入的 skills 相关上下文项。"""
        return items

    def build_skill_messages(self) -> list[ModelMessage]:
        """返回需要注入到渲染结果中的 skills 消息。"""
        return []