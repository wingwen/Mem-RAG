"""DashScope / 通义千问模型工厂。"""

import os

from langchain_community.chat_models import ChatTongyi

from app.core import config_data as config

os.environ.setdefault("DASHSCOPE_API_KEY", config.DASHSCOPE_API_KEY)


def get_chat_model(*, streaming: bool = False) -> ChatTongyi:
    return ChatTongyi(model=config.CHAT_MODEL, streaming=streaming)


def get_light_chat_model() -> ChatTongyi:
    return ChatTongyi(model=config.CHAT_MODEL_LIGHT)
