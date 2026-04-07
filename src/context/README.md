# Context 模块设计说明

## 定位

`src.context` 负责管理输入到模型之前的上下文状态，并通过 `render()` 统一产出 `ModelRequest`。

- Context 不直接调用 provider。
- Provider 负责消费 `ModelRequest` 并发起模型请求。
- tools、skills 等横切能力通过 mixin 接入，而不是写死在基础 context 中。
- Context 内部使用 `ContextEntry` 表达语义材料，provider 使用 `ModelMessage` 表达最终传输消息。
- `ContextEntry.transport` 用于无损保存 provider 原生消息字段，避免 provider 协议细节直接污染 context 语义层。
- provider 原生字段统一通过 `ContextTransport` 访问，不再作为 `ContextEntry` 顶层字段暴露。
- `render()` 是唯一标准输出协议；`render_messages()` 与 `render_tools()` 只是读取最终渲染结果的便利接口。

## 当前组件

- `BaseContext`
  - 定义上下文项生命周期与渲染协议。
  - 提供入库前、渲染前、渲染后等扩展钩子。
  - `add_response_message()` / `add_response_messages()` 用于以 provider 响应语义回填消息。
- `ContextEntry`
  - Context 内部项结构。
  - 顶层保存 `kind`、`metadata`、`priority`、`pinned` 等 context 语义信息。
  - provider 往返需要保留的字段收敛在 `transport` 中，而不是继续膨胀顶层字段。
- `ContextTransport`
  - `ContextEntry.transport` 使用的传输态结构。
  - 承载 `name`、`tool_calls`、`tool_call_id`、`extra` 等 provider 协议细节。
- `ConversationBufferContext`
  - 最小可用的 buffer 实现。
  - 只保存上下文项，不做裁剪。
- `BufferContext`
  - `ConversationBufferContext` 的短别名。
- `ToolsContextMixin` / `SkillsContextMixin`
  - 提供 tools/skills 的渲染增强入口。
  - 默认建议在 ContextEntry 语义层扩展；最终消息阶段的注入仅用于兼容既有 prompt 结构。
- `MessageMetadataMixin`
  - 预留上下文项 metadata 增强入口。
- `tools`
  - 提供函数到 JSON Schema 的转换、工具抽象、注册表，以及 `bash` / `read` / `write` 三个内置工具。

## 使用方式

```python
from src.context import ContextEntry, ConversationBufferContext
from src.provider import ModelMessage

context = ConversationBufferContext()
context.add_item(ContextEntry(role="system", text="你是一个助手", kind="system"))
context.add_message(ModelMessage(role="user", content="你好"))

request = context.render()
messages = request.messages
tools = request.tools
```

当消息来自 provider 响应时，推荐使用 `add_response_message()` 明确表示“回填到 context”，该入口会自动附加 provider 来源语义。

## Tools

可以把普通 Python 函数直接注册成工具：

```python
from src.context import ConversationBufferContext, ToolsContextMixin, tool
from src.provider import ModelMessage


@tool
def add(a: int, b: int) -> int:
  """Add two numbers.

  Args:
    a: First number.
    b: Second number.
  """
  return a + b


class ToolContext(ToolsContextMixin, ConversationBufferContext):
  pass


context = ToolContext(tools=[add])
context.add_message(ModelMessage(role="user", content="请计算 1 + 2"))

request = context.render()
messages = request.messages
tool_schemas = request.tools
result = context.invoke_tool("add", a=1, b=2)
```

`render_tools()` 等价于 `context.render().tools`，用于只读取最终渲染后的工具 schema。

内置工具也在 `src.context.tools` 中提供：

- `ReadTool(base_dir=...)`
- `WriteTool(base_dir=...)`
- `BashTool(base_dir=...)`

`ReadTool` 和 `WriteTool` 会校验目标路径必须位于 `base_dir` 内，避免目录逃逸。
`BashTool` 在 Windows 下会自动使用 PowerShell 执行命令。

## 推荐扩展方式

1. 如果要增强单条上下文项，在 `MessageTransformMixin` 或 `MessageMetadataMixin` 上扩展。
2. 如果要注册自定义工具，优先使用 `@tool` 或 `FunctionTool`，再通过 `ToolsContextMixin.register_tool()` 接入。
3. `render_messages()` 和 `render_tools()` 是辅助接口；主协议始终是 `render() -> ModelRequest`。
4. `tool_render_mode="native"` 是标准路径；`tool_render_mode="prompt"` 仅用于兼容必须通过消息注入工具描述的场景。
5. 如果要做窗口裁剪或 token 控制，优先新增新的 context 类型或策略对象，不要直接修改基础 buffer 行为。