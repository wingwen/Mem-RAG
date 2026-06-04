import datetime
import uuid
import re
import json
import asyncio

from fastapi import FastAPI, Response, HTTPException, Depends, Cookie, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, func, desc
from fastapi.staticfiles import StaticFiles
import os

from app.core.rag import RagService
from app.core.knowledge_base import KnowledgeBaseService
from app.llm.factory import get_light_chat_model
from langchain_core.messages import HumanMessage
from app.core import config_data as config
from app.models.models import Base, User, ChatSession, ChatMessage, MemoryTopic
from app.core.logger import logger
from app.core.prompts import title_generation_prompt, summary_generation_prompt
from app.core.structured_memory import get_structured_memory
from app.core.security import hash_password, verify_password, needs_rehash
from app.utils.memory_migration import ensure_memory_schema
from app.api.database import async_engine, AsyncSessionLocal, get_db
from app.api.schemas import AuthRequest
from app.api.deps import get_current_user, get_owned_session

# --- 1. 配置与初始化 ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_origin_regex=(
        config.CORS_ORIGIN_REGEX if config.CORS_ALLOW_LOCALHOST_ANY_PORT else None
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app.mount("/html", StaticFiles(directory=os.path.join(project_root, "html")), name="html")

rag_service = RagService()
kb_service = KnowledgeBaseService()


def _set_session_cookie(response: Response, cookie_value: str) -> None:
    response.set_cookie(
        key="session_id",
        value=cookie_value,
        httponly=True,
        samesite=config.COOKIE_SAMESITE,
        secure=config.COOKIE_SECURE,
        path="/",
    )


async def save_chat_history(s_id: int, user_in: str, raw_out: str):
    """提取数据并生成标题/总结，存入数据库"""
    logger.info(f"[Backend] 开始处理会话 {s_id} 的后台存储任务...")
    async with AsyncSessionLocal() as db:
        try:
            codes = re.findall(r"```[a-zA-Z0-9\+\#]*\n(.*?)\n```", raw_out, re.DOTALL)
            code_str = "\n---\n".join(codes) if codes else ""
            clean_text = re.sub(r"```.*?```", "", raw_out, flags=re.DOTALL).strip()

            chat_model = get_light_chat_model()

            count_res = await db.execute(select(func.count(ChatMessage.id)).where(ChatMessage.session_id == s_id))
            msg_count = count_res.scalar()

            if msg_count == 0:
                logger.info(f"[Backend] 检测到第一条消息，正在生成标题...")
                try:
                    t_resp = await chat_model.ainvoke([HumanMessage(
                        content=title_generation_prompt.format(user_input=user_in))])
                    new_title = t_resp.content.strip().replace("\u201c", "").replace("\u201d", "").replace("标题：", "")
                    stmt = (
                        update(ChatSession)
                        .where(ChatSession.id == s_id)
                        .values(title=new_title)
                    )
                    await db.execute(stmt)
                    logger.info(f"[Backend] 标题已成功更新为: {new_title}")
                except Exception as e:
                    logger.error(f"[Backend] 标题生成过程出错: {str(e)}")

            summary = ""
            try:
                s_resp = await chat_model.ainvoke([HumanMessage(content=summary_generation_prompt.format(content=clean_text))])
                summary = s_resp.content.strip()
            except Exception as e:
                logger.error(f"[Backend] 生成总结过程出错: {str(e)}")
                summary = clean_text[:50]

            new_msg = ChatMessage(
                session_id=s_id,
                user_input=user_in,
                raw_output=raw_out,
                output_uncode=clean_text,
                code=code_str,
                streamline_input=summary
            )
            db.add(new_msg)
            await db.flush()
            msg_id = new_msg.id
            await db.commit()
            logger.info(f"[Backend] 会话 {s_id} 数据存储完成。")

            asyncio.create_task(asyncio.to_thread(
                get_structured_memory().attach_message_to_topic,
                s_id,
                msg_id,
                user_in,
            ))

        except Exception as e:
            await db.rollback()
            logger.error(f"[Backend] 存储任务发生严重错误: {str(e)}")


@app.on_event("startup")
async def start_event():
    if not config.DASHSCOPE_API_KEY:
        logger.warning("[Config] 未设置 DASHSCOPE_API_KEY，请在 .env 中配置")
    await asyncio.to_thread(ensure_memory_schema)
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.post("/auth/register")
async def register(body: AuthRequest, db: AsyncSession = Depends(get_db)):
    try:
        res = await db.execute(select(User).where(User.username == body.username))
        if res.scalars().first():
            raise HTTPException(status_code=400, detail="用户名已被占用")
        db.add(User(username=body.username, hashed_password=hash_password(body.password)))
        await db.commit()
        return {"status": "success", "message": "注册成功"}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="服务器错误")


@app.post("/auth/login")
async def login(body: AuthRequest, response: Response, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).where(User.username == body.username))
    user = res.scalars().first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    if needs_rehash(user.hashed_password):
        user.hashed_password = hash_password(body.password)

    new_cookie = str(uuid.uuid4())
    user.last_cookie = new_cookie
    await db.commit()
    _set_session_cookie(response, new_cookie)
    return {"status": "success", "message": "登录成功", "data": {"username": user.username}}


@app.post("/sessions")
async def create_session(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    new_uuid = str(uuid.uuid4())
    new_s = ChatSession(session_uuid=new_uuid, user_id=user.id)
    db.add(new_s)
    await db.commit()
    return {"status": "success", "data": {"session_id": new_uuid, "title": "新对话"}}


@app.get("/sessions")
async def get_sessions(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user.id)
        .order_by(desc(ChatSession.update_time))
    )
    return {
        "status": "success",
        "data": [
            {
                "session_id": s.session_uuid,
                "title": s.title,
                "update_time": s.update_time.strftime("%m-%d %H:%M"),
            }
            for s in res.scalars().all()
        ],
    }


@app.get("/sessions/{session_uuid}/memory")
async def get_session_memory(
    curr: ChatSession = Depends(get_owned_session),
    db: AsyncSession = Depends(get_db),
):
    topic_res = await db.execute(
        select(MemoryTopic)
        .where(MemoryTopic.session_id == curr.id)
        .order_by(desc(MemoryTopic.update_time))
    )
    topics = topic_res.scalars().all()
    return {
        "status": "success",
        "data": {
            "active_topic_id": curr.active_topic_id,
            "topics": [
                {
                    "id": t.id,
                    "name": t.topic_name,
                    "summary": t.topic_summary,
                    "keywords": json.loads(t.keywords) if t.keywords else [],
                    "message_count": t.message_count,
                    "is_active": bool(t.is_active),
                }
                for t in topics
            ],
        },
    }


@app.get("/chat/{session_uuid}")
async def get_history(
    curr: ChatSession = Depends(get_owned_session),
    db: AsyncSession = Depends(get_db),
):
    msg_res = await db.execute(
        select(ChatMessage).where(ChatMessage.session_id == curr.id).order_by(ChatMessage.create_time)
    )
    return {
        "status": "success",
        "data": [
            {"user_input": m.user_input, "raw_output": m.raw_output}
            for m in msg_res.scalars().all()
        ],
    }


@app.post("/chat")
async def chat_stream(
    session_uuid: str = Body(..., embed=True),
    input_text: str = Body(..., embed=True),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(ChatSession).where(
            ChatSession.session_uuid == session_uuid,
            ChatSession.user_id == user.id,
        )
    )
    curr = res.scalars().first()
    if not curr:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")

    async def event_generator():
        full_out = ""
        yield "[状态] 正在处理您的问题...\n"
        await asyncio.sleep(0)

        async for chunk in rag_service.astream_response(input_text, session_uuid):
            if chunk.startswith("[状态]"):
                yield chunk if chunk.endswith("\n") else f"{chunk}\n"
                await asyncio.sleep(0)
                continue
            full_out += chunk
            yield chunk
            await asyncio.sleep(0)

        asyncio.create_task(save_chat_history(curr.id, input_text, full_out))

    return StreamingResponse(
        event_generator(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/kb/formats")
async def kb_supported_formats(_user: User = Depends(get_current_user)):
    return {
        "status": "success",
        "data": {
            "formats": kb_service.supported_formats(),
            "chunk_strategy": config.CHUNK_STRATEGY,
            "max_chunk_size": config.MAX_SPLIT_CHAR_NUMBER,
        },
    }


@app.post("/kb/upload")
async def kb_upload(
    file: UploadFile = File(...),
    _user: User = Depends(get_current_user),
):
    filename = file.filename or "unknown"
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="文件内容为空")

    result = await asyncio.to_thread(kb_service.upload_by_bytes, file_bytes, filename)
    if result.startswith("【失败】"):
        raise HTTPException(status_code=400, detail=result.replace("【失败】", ""))

    return {
        "status": "success" if result.startswith("【成功】") else "skipped",
        "message": result,
        "filename": filename,
    }


@app.delete("/delete/{session_uuid}")
async def delete_s(
    curr: ChatSession = Depends(get_owned_session),
    db: AsyncSession = Depends(get_db),
):
    await db.delete(curr)
    await db.commit()
    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
