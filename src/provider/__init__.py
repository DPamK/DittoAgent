"""src.provider — LLM Provider 模块公共接口"""

from .base import BaseProvider, LLMResponse, ModelMessage, ModelRequest, ProviderConfig
from .sync_base import SyncBaseProvider
from .async_base import AsyncBaseProvider

__all__ = [
    # 数据模型
    "ModelMessage",
    "ModelRequest",
    "ProviderConfig",
    "LLMResponse",
    # 抽象基类
    "BaseProvider",
    "SyncBaseProvider",
    "AsyncBaseProvider",
]

try:
    from .openai_provider import AsyncOpenAIProvider, OpenAIProvider
except ImportError:
    AsyncOpenAIProvider = None
    OpenAIProvider = None
else:
    __all__.extend([
        "OpenAIProvider",
        "AsyncOpenAIProvider",
    ])
