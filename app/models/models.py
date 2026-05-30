# 数据库模型定义

import datetime
from sqlalchemy import String, Integer, Text, ForeignKey, DateTime, func, desc
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    create_time: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    update_time: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    last_cookie: Mapped[str] = mapped_column(String(255), nullable=True)
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_uuid: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(255), default="新对话")
    active_topic_id: Mapped[int | None] = mapped_column(
        ForeignKey("memory_topics.id"), nullable=True
    )
    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    topics = relationship(
        "MemoryTopic",
        back_populates="session",
        cascade="all, delete-orphan",
        foreign_keys="MemoryTopic.session_id",
    )


class MemoryTopic(Base):
    """按主题组织的结构化记忆单元。"""
    __tablename__ = "memory_topics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), nullable=False)
    topic_name: Mapped[str] = mapped_column(String(128), nullable=False)
    topic_summary: Mapped[str] = mapped_column(Text, nullable=True, default="")
    keywords: Mapped[str] = mapped_column(Text, nullable=True, default="[]")
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[int] = mapped_column(Integer, default=1)
    session = relationship(
        "ChatSession",
        back_populates="topics",
        foreign_keys=[session_id],
    )
    messages = relationship("ChatMessage", back_populates="topic")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), nullable=False)
    topic_id: Mapped[int | None] = mapped_column(ForeignKey("memory_topics.id"), nullable=True)
    user_input: Mapped[str] = mapped_column(Text)
    raw_output: Mapped[str] = mapped_column(Text)
    output_uncode: Mapped[str] = mapped_column(Text, nullable=True)
    code: Mapped[str] = mapped_column(Text, nullable=True)
    streamline_input: Mapped[str] = mapped_column(Text, nullable=True)
    session = relationship("ChatSession", back_populates="messages")
    topic = relationship("MemoryTopic", back_populates="messages")
