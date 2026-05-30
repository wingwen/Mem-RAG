"""认证与资源归属校验依赖。"""

from fastapi import Cookie, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api.database import get_db
from app.models.models import ChatSession, User


async def get_current_user(
    session_id: str | None = Cookie(None),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not session_id:
        raise HTTPException(status_code=401, detail="未登录")
    res = await db.execute(select(User).where(User.last_cookie == session_id))
    user = res.scalars().first()
    if not user:
        raise HTTPException(status_code=401, detail="无效会话")
    return user


async def get_owned_session(
    session_uuid: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChatSession:
    res = await db.execute(
        select(ChatSession).where(
            ChatSession.session_uuid == session_uuid,
            ChatSession.user_id == user.id,
        )
    )
    session = res.scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")
    return session
