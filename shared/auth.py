"""Authentication helpers — JWT token management and password hashing."""

import datetime
import bcrypt
import jwt

from shared.config import Config


def hash_password(plain: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_jwt(user_id: str, email: str) -> str:
    """Create a signed JWT token with user_id and email claims."""
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=Config.JWT_EXPIRY_HOURS),
        "iat": datetime.datetime.utcnow(),
    }
    return jwt.encode(payload, Config.JWT_SECRET, algorithm="HS256")


def verify_jwt(token: str) -> dict:
    """Decode and validate a JWT token. Returns {"user_id": ..., "email": ...} or raises."""
    payload = jwt.decode(token, Config.JWT_SECRET, algorithms=["HS256"])
    user_id = payload.get("user_id")
    email = payload.get("email")
    if not user_id or not email:
        raise ValueError("JWT payload missing required claims: user_id and email")
    return {"user_id": user_id, "email": email}
