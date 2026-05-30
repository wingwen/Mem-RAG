"""
结构化记忆：按主题分类存储与检索对话历史，提升多轮连贯性。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy import create_engine, select, update, func
from sqlalchemy.orm import sessionmaker

from app.core import config_data as config
from app.core.logger import logger
from app.core.prompts import (
    topic_classification_prompt,
    topic_summary_update_prompt,
)
from app.models.models import Base, ChatMessage, ChatSession, MemoryTopic

sync_engine = create_engine(
    config.ASYNC_DATABASE_URL.replace("mysql+aiomysql://", "mysql+pymysql://"),
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)


@dataclass
class MemoryContext:
    """构建好的结构化记忆上下文。"""
    active_topic: str
    active_topic_id: int | None
    memory_block: str
    history_messages: list[BaseMessage]


class StructuredMemoryService:
    def __init__(self):
        self._chat_model = ChatTongyi(model="qwen-turbo")

    def _get_db_session_id(self, session_uuid: str) -> int | None:
        db = SessionLocal()
        try:
            return db.execute(
                select(ChatSession.id).where(ChatSession.session_uuid == session_uuid)
            ).scalars().first()
        finally:
            db.close()

    def _load_topics(self, db, session_id: int) -> list[MemoryTopic]:
        return list(
            db.execute(
                select(MemoryTopic)
                .where(MemoryTopic.session_id == session_id)
                .order_by(MemoryTopic.update_time.desc())
            ).scalars().all()
        )

    def _load_messages(self, db, session_id: int) -> list[ChatMessage]:
        return list(
            db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.create_time)
            ).scalars().all()
        )

    @staticmethod
    def _parse_keywords(raw: str | None) -> list[str]:
        if not raw:
            return []
        try:
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return [k.strip() for k in raw.split(",") if k.strip()]

    @staticmethod
    def _keyword_overlap(query: str, keywords: list[str]) -> float:
        if not keywords:
            return 0.0
        q_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]{2,}", query.lower()))
        if not q_tokens:
            return 0.0
        hits = sum(1 for kw in keywords if kw.lower() in query.lower() or kw in query)
        return hits / max(len(keywords), 1)

    def _classify_topic_fast(
        self, query: str, topics: list[MemoryTopic]
    ) -> MemoryTopic | None:
        best_topic = None
        best_score = 0.0
        for topic in topics:
            keywords = self._parse_keywords(topic.keywords)
            score = self._keyword_overlap(query, keywords + [topic.topic_name])
            if score > best_score:
                best_score = score
                best_topic = topic
        if best_score >= config.MEMORY_TOPIC_MATCH_THRESHOLD:
            return best_topic
        return None

    def _classify_topic_llm(
        self, query: str, topics: list[MemoryTopic]
    ) -> tuple[str, list[str], bool]:
        """返回 (topic_name, keywords, is_new_topic)。"""
        topic_lines = "\n".join(
            f"- {t.topic_name}（关键词: {', '.join(self._parse_keywords(t.keywords))}）"
            for t in topics
        ) or "（暂无历史主题）"

        prompt = topic_classification_prompt.format(
            user_input=query,
            existing_topics=topic_lines,
        )
        try:
            resp = self._chat_model.invoke([HumanMessage(content=prompt)])
            text = resp.content.strip()
            if text.startswith("```"):
                text = re.sub(r"```json?\n?|\n?```", "", text).strip()
            data = json.loads(text)
            return (
                data.get("topic_name", "通用对话")[:64],
                data.get("keywords", [])[:10],
                bool(data.get("is_new_topic", True)),
            )
        except Exception as exc:
            logger.warning(f"[Memory] 主题分类 LLM 失败，使用启发式: {exc}")
            return query[:20] or "通用对话", [], True

    def resolve_topic(
        self,
        session_uuid: str,
        user_input: str,
        use_llm: bool = False,
        create_if_missing: bool = True,
    ) -> MemoryTopic | None:
        """为当前用户输入解析/创建主题。"""
        db = SessionLocal()
        try:
            session_id = self._get_db_session_id(session_uuid)
            if not session_id:
                return None

            session_row = db.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            ).scalars().first()

            topics = self._load_topics(db, session_id)
            matched = self._classify_topic_fast(user_input, topics)

            if matched:
                db.execute(
                    update(MemoryTopic)
                    .where(MemoryTopic.id == matched.id)
                    .values(is_active=1)
                )
                db.execute(
                    update(ChatSession)
                    .where(ChatSession.id == session_id)
                    .values(active_topic_id=matched.id)
                )
                db.commit()
                return matched

            if session_row and session_row.active_topic_id:
                active = db.execute(
                    select(MemoryTopic).where(MemoryTopic.id == session_row.active_topic_id)
                ).scalars().first()
                if active:
                    return active

            if not create_if_missing:
                return None

            if use_llm or len(topics) < config.MEMORY_MAX_TOPICS:
                name, keywords, is_new = self._classify_topic_llm(user_input, topics)
                if not is_new:
                    for t in topics:
                        if t.topic_name == name:
                            db.execute(
                                update(MemoryTopic)
                                .where(MemoryTopic.id == t.id)
                                .values(is_active=1)
                            )
                            db.execute(
                                update(ChatSession)
                                .where(ChatSession.id == session_id)
                                .values(active_topic_id=t.id)
                            )
                            db.commit()
                            return t

                new_topic = MemoryTopic(
                    session_id=session_id,
                    topic_name=name,
                    topic_summary="",
                    keywords=json.dumps(keywords, ensure_ascii=False),
                    message_count=0,
                    is_active=1,
                )
                db.add(new_topic)
                db.flush()

                db.execute(
                    update(ChatSession)
                    .where(ChatSession.id == session_id)
                    .values(active_topic_id=new_topic.id)
                )
                db.commit()
                db.refresh(new_topic)
                return new_topic

            if topics:
                active = topics[0]
                db.execute(
                    update(ChatSession)
                    .where(ChatSession.id == session_id)
                    .values(active_topic_id=active.id)
                )
                db.commit()
                return active
            return None
        finally:
            db.close()

    def build_context(self, session_uuid: str, current_query: str) -> MemoryContext:
        """构建结构化记忆块 + 精选历史消息。"""
        db = SessionLocal()
        try:
            session_id = self._get_db_session_id(session_uuid)
            if not session_id:
                return MemoryContext("新对话", None, "", [])

            topics = self._load_topics(db, session_id)
            messages = self._load_messages(db, session_id)
            active_topic = self.resolve_topic(
                session_uuid, current_query, use_llm=False, create_if_missing=False
            )

            active_topic_id = active_topic.id if active_topic else None
            active_topic_name = active_topic.topic_name if active_topic else "新对话"

            memory_lines = ["【结构化对话记忆】"]

            if topics:
                memory_lines.append("■ 会话主题索引：")
                for t in topics[: config.MEMORY_MAX_TOPICS]:
                    kw = ", ".join(self._parse_keywords(t.keywords)[:5])
                    marker = " ← 当前" if t.id == active_topic_id else ""
                    summary = (t.topic_summary or "（暂无摘要）")[:120]
                    memory_lines.append(
                        f"  · [{t.topic_name}]{marker} 关键词:{kw} | 摘要:{summary}"
                    )

            same_topic_msgs = [
                m for m in messages if m.topic_id == active_topic_id
            ] if active_topic_id else []

            if same_topic_msgs:
                memory_lines.append(f"\n■ 当前主题「{active_topic_name}」近期要点：")
                for m in same_topic_msgs[-config.MEMORY_SAME_TOPIC_TURNS :]:
                    brief = m.streamline_input or m.output_uncode or m.raw_output
                    memory_lines.append(f"  - 用户: {m.user_input[:80]}")
                    memory_lines.append(f"    助手: {(brief or '')[:100]}")

            other_topics = [t for t in topics if t.id != active_topic_id]
            if other_topics:
                memory_lines.append("\n■ 其他主题概要（供跨主题参考）：")
                for t in other_topics[:3]:
                    if t.topic_summary:
                        memory_lines.append(f"  · {t.topic_name}: {t.topic_summary[:80]}")

            memory_block = "\n".join(memory_lines)

            history_messages = self._select_history_messages(
                messages, active_topic_id, current_query
            )

            return MemoryContext(
                active_topic=active_topic_name,
                active_topic_id=active_topic_id,
                memory_block=memory_block,
                history_messages=history_messages,
            )
        finally:
            db.close()

    def _select_history_messages(
        self,
        messages: list[ChatMessage],
        active_topic_id: int | None,
        current_query: str,
    ) -> list[BaseMessage]:
        """精选多轮历史：同主题优先 + 最近全局轮次。"""
        if not messages:
            return []

        selected_ids: set[int] = set()
        result_pairs: list[tuple[ChatMessage, str]] = []

        same_topic = [m for m in messages if m.topic_id == active_topic_id]
        for m in same_topic[-config.MEMORY_SAME_TOPIC_TURNS :]:
            if m.id not in selected_ids:
                selected_ids.add(m.id)
                content = m.output_uncode or m.raw_output or ""
                if m != same_topic[-1]:
                    content = m.streamline_input or content[:200]
                result_pairs.append((m, content))

        recent_any = messages[-config.MEMORY_MAX_RECENT_TURNS :]
        for m in recent_any:
            if m.id not in selected_ids:
                selected_ids.add(m.id)
                content = m.streamline_input or m.output_uncode or m.raw_output or ""
                result_pairs.append((m, content[:200]))

        result_pairs.sort(key=lambda x: x[0].create_time)

        langchain_msgs: list[BaseMessage] = []
        for m, ai_content in result_pairs:
            langchain_msgs.append(HumanMessage(content=m.user_input))
            langchain_msgs.append(AIMessage(content=ai_content))

        return langchain_msgs

    def attach_message_to_topic(
        self, session_db_id: int, message_id: int, user_input: str
    ) -> None:
        """消息入库后：绑定主题并更新主题摘要（后台调用）。"""
        db = SessionLocal()
        try:
            session_row = db.execute(
                select(ChatSession).where(ChatSession.id == session_db_id)
            ).scalars().first()
            if not session_row:
                return

            topics = self._load_topics(db, session_db_id)
            topic = self._classify_topic_fast(user_input, topics)

            if not topic:
                name, keywords, _ = self._classify_topic_llm(user_input, topics)
                topic = MemoryTopic(
                    session_id=session_db_id,
                    topic_name=name,
                    topic_summary="",
                    keywords=json.dumps(keywords, ensure_ascii=False),
                    message_count=0,
                    is_active=1,
                )
                db.add(topic)
                db.flush()

            msg = db.execute(
                select(ChatMessage).where(ChatMessage.id == message_id)
            ).scalars().first()
            if msg:
                msg.topic_id = topic.id

            topic_messages = db.execute(
                select(ChatMessage)
                .where(
                    ChatMessage.session_id == session_db_id,
                    ChatMessage.topic_id == topic.id,
                )
                .order_by(ChatMessage.create_time)
            ).scalars().all()

            turns_text = "\n".join(
                f"用户:{m.user_input}\n助手:{m.streamline_input or m.output_uncode or ''}"
                for m in topic_messages[-5:]
            )
            try:
                resp = self._chat_model.invoke([
                    HumanMessage(content=topic_summary_update_prompt.format(
                        topic_name=topic.topic_name,
                        conversation_turns=turns_text,
                    ))
                ])
                topic.topic_summary = resp.content.strip()[:300]
            except Exception as exc:
                logger.warning(f"[Memory] 主题摘要更新失败: {exc}")

            topic.message_count = len(topic_messages)
            topic.is_active = 1

            db.execute(
                update(ChatSession)
                .where(ChatSession.id == session_db_id)
                .values(active_topic_id=topic.id)
            )
            db.execute(
                update(MemoryTopic)
                .where(MemoryTopic.session_id == session_db_id, MemoryTopic.id != topic.id)
                .values(is_active=0)
            )
            db.commit()
            logger.info(f"[Memory] 消息 {message_id} 归入主题「{topic.topic_name}」")
        except Exception as exc:
            db.rollback()
            logger.error(f"[Memory] 主题绑定失败: {exc}")
        finally:
            db.close()


_memory_service: StructuredMemoryService | None = None


def get_structured_memory() -> StructuredMemoryService:
    global _memory_service
    if _memory_service is None:
        _memory_service = StructuredMemoryService()
    return _memory_service
