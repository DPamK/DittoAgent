from src.context import (
    ContextEntry,
    ContextTransport,
    ConversationBufferContext,
    MessageMetadataMixin,
    SkillsContextMixin,
    ToolsContextMixin,
)
from src.provider.base import ModelMessage


def test_conversation_buffer_context_keeps_message_order() -> None:
    context = ConversationBufferContext()
    context.add_message(ModelMessage(role="system", content="system prompt"))
    context.add_message(ModelMessage(role="user", content="hello"))

    messages = context.render_messages()

    assert [message.role for message in messages] == ["system", "user"]
    assert [message.content for message in messages] == ["system prompt", "hello"]


def test_render_returns_isolated_message_copies() -> None:
    context = ConversationBufferContext(
        items=[ContextEntry(role="user", text="hello", metadata={"source": "input"})]
    )

    rendered = context.render_messages()
    rendered[0].extra["source"] = "changed"
    rendered.append(ModelMessage(role="assistant", content="new"))

    rerendered = context.render_messages()

    assert len(rerendered) == 1
    assert rerendered[0].content == "hello"


def test_context_keeps_internal_metadata_separate_from_provider_messages() -> None:
    context = ConversationBufferContext(
        items=[ContextEntry(role="user", text="hello", metadata={"source": "input"})]
    )

    assert context.items[0].metadata["source"] == "input"
    assert context.render_messages()[0].to_dict() == {"role": "user", "content": "hello"}


def test_context_entry_transport_round_trips_provider_fields() -> None:
    message = ModelMessage(
        role="assistant",
        content="",
        name="planner",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "write", "arguments": "{}"},
            }
        ],
        extra={"provider": "openai"},
    )

    entry = ContextEntry.from_message(message, metadata={"source": "provider"})
    rerendered = entry.to_message()

    assert entry.metadata == {"source": "provider"}
    assert entry.transport == ContextTransport(
        name="planner",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "write", "arguments": "{}"},
            }
        ],
        extra={"provider": "openai"},
    )
    assert rerendered.name == "planner"
    assert rerendered.tool_calls[0]["id"] == "call_1"
    assert rerendered.extra == {"provider": "openai"}


def test_context_entry_uses_transport_as_the_only_protocol_state() -> None:
    entry = ContextEntry(
        role="tool",
        text="ok",
        transport=ContextTransport(
            name="write",
            tool_calls=[{"id": "call_1"}],
            tool_call_id="call_1",
            extra={"raw": True},
        ),
    )

    assert entry.transport.name == "write"
    assert entry.transport.tool_calls == [{"id": "call_1"}]
    assert entry.transport.tool_call_id == "call_1"
    assert entry.transport.extra == {"raw": True}


def test_add_response_message_round_trips_provider_message() -> None:
    context = ConversationBufferContext()

    context.add_response_message(
        ModelMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "write", "arguments": "{}"},
                }
            ],
        )
    )

    item = context.items[0]
    assert item.kind == "provider_response"
    assert item.metadata["source"] == "provider"
    assert item.transport.tool_calls[0]["id"] == "call_1"
    assert context.render().messages[0].tool_calls[0]["id"] == "call_1"


def test_add_message_keeps_generic_message_semantics() -> None:
    context = ConversationBufferContext()

    context.add_message(ModelMessage(role="assistant", content="hello"))

    item = context.items[0]
    assert item.kind == "message"
    assert item.metadata == {}


def test_clear_removes_all_messages() -> None:
    context = ConversationBufferContext(
        messages=[
            ModelMessage(role="system", content="system prompt"),
            ModelMessage(role="user", content="hello"),
        ]
    )

    context.clear()

    assert context.items == ()
    assert context.render_messages() == []


def test_message_metadata_mixin_enriches_stored_messages() -> None:
    class TaggedContext(MessageMetadataMixin, ConversationBufferContext):
        def build_message_metadata(self, item: ContextEntry) -> dict[str, object]:
            return {"length": len(item.text)}

    context = TaggedContext()
    context.add_message(ModelMessage(role="user", content="hello"))

    assert context.items[0].metadata["length"] == 5


def test_render_mixins_are_composable() -> None:
    class ToolAndSkillContext(ToolsContextMixin, SkillsContextMixin, ConversationBufferContext):
        def build_skill_messages(self) -> list[ModelMessage]:
            return [ModelMessage(role="system", content="skills")]

    def echo(text: str) -> str:
        return text

    context = ToolAndSkillContext(tools=[echo], messages=[ModelMessage(role="user", content="hello")])

    rendered = context.render_messages()
    request = context.render()

    assert [message.content for message in rendered] == ["skills", "hello"]
    assert request.tools[0]["function"]["name"] == "echo"