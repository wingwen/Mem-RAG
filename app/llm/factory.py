"""DashScope / 通义千问模型工厂。"""

import os

from langchain_community.chat_models import ChatTongyi

from app.core import config_data as config

os.environ.setdefault("DASHSCOPE_API_KEY", config.DASHSCOPE_API_KEY)


def _model_extra_kwargs(model_name: str) -> dict:
    """qwen3 系列非流式调用须显式关闭 enable_thinking。"""
    if "qwen3" in (model_name or "").lower():
        return {"enable_thinking": False}
    return {}


def get_chat_model(*, streaming: bool = False) -> ChatTongyi:
    model = config.CHAT_MODEL
    return ChatTongyi(
        model=model,
        streaming=streaming,
        model_kwargs=_model_extra_kwargs(model),
    )


def get_light_chat_model() -> ChatTongyi:
    model = config.CHAT_MODEL_LIGHT
    return ChatTongyi(
        model=model,
        model_kwargs=_model_extra_kwargs(model),
    )
