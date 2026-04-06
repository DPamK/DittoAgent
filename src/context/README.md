# Context 模块设计说明

## 定位

`src.context` 负责管理输入到模型之前的上下文状态，并通过 `render()` 统一产出 `ModelRequest`。

- Context 不直接调用 provider。
- Provider 负责消费 `ModelRequest` 并发起模型请求。
- tools、skills 等横切能力通过 mixin 接入，而不是写死在基础 context 中。
- Context 内部使用 `ContextEntry` 表达语义材料，provider 使用 `ModelMessage` 表达最终传输消息。

## 当前组件

- `BaseContext`
  - 定义上下文项生命周期与渲染协议。
  - 提供入库前、渲染前、渲染后等扩展钩子。
- `ContextEntry`
  - Context 内部项结构。
  - 保存 `kind`、`metadata`、`priority`、`pinned` 等 provider 无需感知的语义信息。
- `ConversationBufferContext`
  - 最小可用的 buffer 实现。
  - 只保存上下文项，不做裁剪。
- `ToolsContextMixin` / `SkillsContextMixin`
  - 提供 tools/skills 的渲染增强入口。
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
```

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

内置工具也在 `src.context.tools` 中提供：

- `ReadTool(base_dir=...)`
- `WriteTool(base_dir=...)`
- `BashTool(base_dir=...)`

`ReadTool` 和 `WriteTool` 会校验目标路径必须位于 `base_dir` 内，避免目录逃逸。
`BashTool` 在 Windows 下会自动使用 PowerShell 执行命令。

## 推荐扩展方式

1. 如果要增强单条上下文项，在 `MessageTransformMixin` 或 `MessageMetadataMixin` 上扩展。
2. 如果要注册自定义工具，优先使用 `@tool` 或 `FunctionTool`，再通过 `ToolsContextMixin.register_tool()` 接入。
3. `render_messages()` 和 `render_tools()` 是辅助接口；主协议是 `render() -> ModelRequest`。
4. 如果要做窗口裁剪或 token 控制，优先新增新的 context 类型或策略对象，不要直接修改基础 buffer 行为。