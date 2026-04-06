import os

import pytest

from src.context import (
    BashTool,
    ConversationBufferContext,
    FunctionTool,
    ReadTool,
    ToolRegistry,
    ToolsContextMixin,
    WriteTool,
    function_to_json_schema,
)
from src.provider.base import ModelMessage


def test_function_to_json_schema_uses_annotations_and_docstring() -> None:
    def sample_tool(query: str, limit: int = 5, tags: list[str] | None = None) -> str:
        """Search something.

        Args:
            query: Query text.
            limit: Max result count.
            tags: Optional tag filters.
        """

    schema = function_to_json_schema(sample_tool)

    assert schema["function"]["name"] == "sample_tool"
    assert schema["function"]["description"] == "Search something."
    parameters = schema["function"]["parameters"]
    assert parameters["required"] == ["query"]
    assert parameters["properties"]["query"]["type"] == "string"
    assert parameters["properties"]["limit"]["default"] == 5
    assert parameters["properties"]["tags"]["type"] == "array"


def test_function_tool_invokes_function_and_exports_schema() -> None:
    def multiply(left: int, right: int) -> int:
        """Multiply two numbers."""
        return left * right

    tool = FunctionTool(multiply)
    result = tool.invoke(left=2, right=4)

    assert tool.to_schema()["function"]["name"] == "multiply"
    assert result.status == "success"
    assert result.data == 8


def test_tool_registry_rejects_duplicate_names() -> None:
    def greet(name: str) -> str:
        return f"hi {name}"

    registry = ToolRegistry()
    registry.register(FunctionTool(greet))

    with pytest.raises(ValueError):
        registry.register(FunctionTool(greet))


def test_tools_context_mixin_registers_tools_and_renders_message() -> None:
    def echo(text: str) -> str:
        """Echo text."""
        return text

    class ToolContext(ToolsContextMixin, ConversationBufferContext):
        pass

    context = ToolContext(tools=[echo], messages=[ModelMessage(role="user", content="hello")])

    request = context.render()

    assert context.get_tool("echo").name == "echo"
    assert context.render_tools()[0]["function"]["name"] == "echo"
    assert request.messages[0].role == "user"
    assert request.tools[0]["function"]["name"] == "echo"
    assert context.invoke_tool("echo", text="ping").data == "ping"


def test_read_and_write_tools_enforce_base_dir(tmp_path) -> None:
    writer = WriteTool(base_dir=tmp_path)
    reader = ReadTool(base_dir=tmp_path)

    write_result = writer.invoke(path="notes/hello.txt", content="line1\nline2")
    read_result = reader.invoke(path="notes/hello.txt", start_line=2)
    denied_result = writer.invoke(path="../outside.txt", content="blocked")

    assert write_result.status == "success"
    assert read_result.status == "success"
    assert read_result.data["content"] == "line2"
    assert denied_result.status == "error"


def test_bash_tool_runs_command(tmp_path) -> None:
    tool = BashTool(base_dir=tmp_path)
    command = "Write-Output hello" if os.name == "nt" else "printf hello"

    result = tool.invoke(command=command, timeout=5)

    assert result.status == "success"
    assert "hello" in result.data["stdout"]