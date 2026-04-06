"""Tools 抽象、schema 转换与注册表。"""

from __future__ import annotations

import inspect
import json
import re
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Union, get_args, get_origin, get_type_hints


def _parse_docstring(docstring: str | None) -> tuple[str, dict[str, str]]:
    if not docstring:
        return "", {}

    lines = [line.rstrip() for line in inspect.cleandoc(docstring).splitlines()]
    if not lines:
        return "", {}

    summary_lines: list[str] = []
    param_docs: dict[str, str] = {}
    in_args_section = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if in_args_section:
                continue
            if summary_lines:
                break
            continue

        lowered = line.lower().rstrip(":")
        if lowered in {"args", "arguments", "parameters"}:
            in_args_section = True
            continue

        if in_args_section:
            match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:\s*(.+)$", line)
            if match:
                param_docs[match.group(1)] = match.group(2).strip()
                continue

            if raw_line.startswith((" ", "\t")) and param_docs:
                last_key = next(reversed(param_docs))
                param_docs[last_key] = f"{param_docs[last_key]} {line}".strip()
                continue

            in_args_section = False

        if not in_args_section:
            summary_lines.append(line)

    return " ".join(summary_lines).strip(), param_docs


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Signature.empty or annotation is Any:
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in {Union, types.UnionType}:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1 and len(non_none_args) != len(args):
            return _annotation_to_schema(non_none_args[0])
        return {"anyOf": [_annotation_to_schema(arg) for arg in non_none_args or args]}

    if origin is Literal:
        literal_values = list(args)
        schema: dict[str, Any] = {"enum": literal_values}
        if literal_values:
            value_type = type(literal_values[0])
            schema.update(_annotation_to_schema(value_type))
        return schema

    if origin in {list, tuple, set}:
        item_schema = _annotation_to_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema}

    if origin is dict:
        value_schema = _annotation_to_schema(args[1]) if len(args) > 1 else {}
        schema: dict[str, Any] = {"type": "object"}
        if value_schema:
            schema["additionalProperties"] = value_schema
        return schema

    if annotation in {str, Path}:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is dict:
        return {"type": "object"}
    if annotation in {list, tuple, set}:
        return {"type": "array"}

    if inspect.isclass(annotation):
        return {"type": "string"}

    return {}


def _parameter_schema(
    parameter: inspect.Parameter,
    annotation: Any,
    description: str | None,
) -> dict[str, Any]:
    schema = _annotation_to_schema(annotation)
    if description:
        schema["description"] = description

    if parameter.default is not inspect.Signature.empty:
        default_value = parameter.default
        if isinstance(default_value, Path):
            default_value = str(default_value)
        if isinstance(default_value, (str, int, float, bool, list, dict)) or default_value is None:
            schema["default"] = default_value

    return schema


def function_parameters_schema(function: Callable[..., Any]) -> dict[str, Any]:
    signature = inspect.signature(function)
    type_hints = get_type_hints(function)
    _, param_docs = _parse_docstring(function.__doc__)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, parameter in signature.parameters.items():
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            continue

        annotation = type_hints.get(name, parameter.annotation)
        properties[name] = _parameter_schema(parameter, annotation, param_docs.get(name))

        if parameter.default is inspect.Signature.empty:
            required.append(name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def function_to_json_schema(
    function: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    summary, _ = _parse_docstring(function.__doc__)
    function_name = name or function.__name__
    function_description = description or summary or f"Call {function_name}"
    return {
        "type": "function",
        "function": {
            "name": function_name,
            "description": function_description,
            "parameters": function_parameters_schema(function),
        },
    }


@dataclass
class ToolResult:
    """工具执行结果。"""

    status: Literal["success", "error"]
    data: Any = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "data": self.data,
            "message": self.message,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_text(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class BaseTool(ABC):
    """所有工具的基础抽象。"""

    def __init__(self, name: str, description: str, *, strict: bool = False) -> None:
        self.name = name
        self.description = description
        self.strict = strict

    @abstractmethod
    def args_schema(self) -> dict[str, Any]:
        """返回工具参数 schema。"""

    def to_schema(self) -> dict[str, Any]:
        schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema(),
            },
        }
        if self.strict:
            schema["function"]["strict"] = True
        return schema

    @abstractmethod
    def invoke(self, **kwargs: Any) -> ToolResult:
        """执行工具。"""


class FunctionTool(BaseTool):
    """将普通 Python 函数包装为工具。"""

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool = False,
    ) -> None:
        schema = function_to_json_schema(function, name=name, description=description)
        function_meta = schema["function"]
        super().__init__(
            name=function_meta["name"],
            description=function_meta["description"],
            strict=strict,
        )
        self._function = function
        self._schema = function_meta["parameters"]

    def args_schema(self) -> dict[str, Any]:
        return dict(self._schema)

    def invoke(self, **kwargs: Any) -> ToolResult:
        try:
            result = self._function(**kwargs)
        except Exception as exc:
            return ToolResult(status="error", message=str(exc))
        return ToolResult(status="success", data=result)


def tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool = False,
) -> Callable[[Callable[..., Any]], FunctionTool] | FunctionTool:
    """将函数包装为 FunctionTool 的装饰器。"""

    def decorator(func: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(func, name=name, description=description, strict=strict)

    if function is None:
        return decorator
    return decorator(function)


class ToolRegistry:
    """工具注册表。"""

    _NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

    def __init__(self, tools: Iterable[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        if tools is not None:
            self.register_many(tools)

    def register(self, tool: BaseTool, *, replace: bool = False) -> BaseTool:
        if not self._NAME_PATTERN.match(tool.name):
            raise ValueError(f"非法工具名: {tool.name}")
        if tool.name in self._tools and not replace:
            raise ValueError(f"工具已存在: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def register_many(self, tools: Iterable[BaseTool], *, replace: bool = False) -> list[BaseTool]:
        return [self.register(tool, replace=replace) for tool in tools]

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"未注册的工具: {name}") from exc

    def list(self) -> list[BaseTool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.to_schema() for tool in self.list()]

    def invoke(self, name: str, **kwargs: Any) -> ToolResult:
        return self.get(name).invoke(**kwargs)


def ensure_tool(tool_or_callable: BaseTool | Callable[..., Any]) -> BaseTool:
    if isinstance(tool_or_callable, BaseTool):
        return tool_or_callable
    if callable(tool_or_callable):
        return FunctionTool(tool_or_callable)
    raise TypeError("tool 必须是 BaseTool 或可调用对象")