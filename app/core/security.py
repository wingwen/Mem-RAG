"""密码哈希与校验（bcrypt + 旧版 SHA256 兼容）。"""

import hashlib

import bcrypt

from app.core.config_data import SALT_SUFFIX

BCRYPT_PREFIX = "$2"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not hashed_password:
        return False
    if hashed_password.startswith(BCRYPT_PREFIX):
        try:
            return bcrypt.checkpw(
                plain_password.encode("utf-8"),
                hashed_password.encode("utf-8"),
            )
        except ValueError:
            return False
    legacy = hashlib.sha256((plain_password + SALT_SUFFIX).encode("utf-8")).hexdigest()
    return legacy == hashed_password


def needs_rehash(hashed_password: str) -> bool:
    return not hashed_password.startswith(BCRYPT_PREFIX)
