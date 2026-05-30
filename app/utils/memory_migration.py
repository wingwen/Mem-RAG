"""结构化记忆相关表结构迁移（兼容已有数据库）。"""

from sqlalchemy import inspect, text

from app.core.logger import logger
from app.models.models import Base

SYNC_URL = None


def _get_sync_engine():
    from sqlalchemy import create_engine
    from app.core import config_data as config

    global SYNC_URL
    if SYNC_URL is None:
        SYNC_URL = config.ASYNC_DATABASE_URL.replace("mysql+aiomysql://", "mysql+pymysql://")
    return create_engine(SYNC_URL, echo=False)


def ensure_memory_schema() -> None:
    engine = _get_sync_engine()
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    if "chat_messages" in inspector.get_table_names():
        cols = {c["name"] for c in inspector.get_columns("chat_messages")}
        if "topic_id" not in cols:
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE chat_messages ADD COLUMN topic_id INT NULL"
                ))
            logger.info("[Migration] chat_messages.topic_id 已添加")

    if "chat_sessions" in inspector.get_table_names():
        cols = {c["name"] for c in inspector.get_columns("chat_sessions")}
        if "active_topic_id" not in cols:
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE chat_sessions ADD COLUMN active_topic_id INT NULL"
                ))
            logger.info("[Migration] chat_sessions.active_topic_id 已添加")

    logger.info("[Migration] 结构化记忆表结构就绪")
