# 数据库存储记忆功能

from typing import Sequence
from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import sessionmaker
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, message_to_dict, messages_from_dict, BaseMessage
from app.core import config_data as config
from app.models.models import Base, ChatMessage, ChatSession
from app.core.logger import logger
from app.core.structured_memory import get_structured_memory

# 初始化数据库连接（使用同步API）
sync_engine = create_engine(config.ASYNC_DATABASE_URL.replace('mysql+aiomysql://', 'mysql+pymysql://'), echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

# 确保数据库表存在
Base.metadata.create_all(bind=sync_engine)


class DatabaseChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id):
        """
        :param session_id: 会话UUID
        """
        self.session_id = session_id

    def _get_session_id(self):
        """获取会话的数据库ID"""
        db = SessionLocal()
        try:
            result = db.execute(
                select(ChatSession.id).where(ChatSession.session_uuid == self.session_id)
            )
            session_id = result.scalars().first()
            return session_id
        finally:
            db.close()

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        保存消息到数据库
        注意：这里我们只处理新消息，因为完整的消息存储逻辑在api_service.py的save_chat_history中
        :param messages: 消息序列
        :return:
        """
        # 实际的消息存储由api_service.py中的save_chat_history处理
        # 这里我们不需要做任何操作，因为消息已经通过API端点保存到数据库
        pass

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """使用结构化记忆服务精选历史消息。"""
        last_query = ""
        db = SessionLocal()
        try:
            session_id = self._get_session_id()
            if not session_id:
                return []
            result = db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.create_time.desc())
                .limit(1)
            )
            last_msg = result.scalars().first()
            if last_msg:
                last_query = last_msg.user_input
        finally:
            db.close()

        ctx = get_structured_memory().build_context(self.session_id, last_query)
        return ctx.history_messages

    def clear(self):
        """
        清除会话的所有消息
        :return:
        """
        db = SessionLocal()
        try:
            session_id = self._get_session_id()
            if session_id:
                db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
                db.commit()
        finally:
            db.close()


def get_history(session_id) -> DatabaseChatMessageHistory:
    messages = DatabaseChatMessageHistory(session_id).messages
    logger.debug(f"[History] 会话 {session_id} 的历史消息: {messages}")
    return DatabaseChatMessageHistory(session_id)

