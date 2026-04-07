"""基础 Buffer Context 实现。"""

from __future__ import annotations

from typing import Iterable

from src.provider.base import ModelMessage

from .base import BaseContext
from .types import ContextEntry


class ConversationBufferContext(BaseContext):
    """最小可用的对话 Buffer Context。

    行为对齐基础版 ConversationBufferMemory：
    - 按顺序保存消息
    - 不做裁剪
    - ``render()`` 直接返回当前消息快照
    """

    def __init__(
        self,
        items: Iterable[ContextEntry] | None = None,
        messages: Iterable[ModelMessage] | None = None,
    ) -> None:
        self._items: list[ContextEntry] = []
        super().__init__(items=items, messages=messages)

    def _store_item(self, item: ContextEntry) -> None:
        self._items.append(item)

    def _list_items(self) -> list[ContextEntry]:
        return self._items

    def _clear_items(self) -> None:
        self._items.clear()



