"""Tests for shared.auth — JWT and bcrypt helpers."""

import time
import datetime
from unittest.mock import patch

import pytest
import jwt as pyjwt

from shared.auth import hash_password, verify_password, create_jwt, verify_jwt
from shared.config import Config


class TestPasswordHashing:
    """Tests for bcrypt hash/verify."""

    def test_hash_password_returns_string(self):
        hashed = hash_password("secret123")
        assert isinstance(hashed, str)
        assert hashed.startswith("$2b$")

    def test_verify_password_correct(self):
        hashed = hash_password("mypassword")
        assert verify_password("mypassword", hashed) is True

    def test_verify_password_wrong(self):
        hashed = hash_password("correctpassword")
        assert verify_password("wrongpassword", hashed) is False

    def test_hash_password_different_each_time(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2  # Different salts


class TestJWT:
    """Tests for JWT creation and verification."""

    def test_create_jwt_returns_string(self):
        token = create_jwt("user-123", "user@example.com")
        assert isinstance(token, str)
        assert len(token) > 20

    def test_verify_jwt_returns_claims(self):
        token = create_jwt("user-456", "test@test.com")
        claims = verify_jwt(token)
        assert claims["user_id"] == "user-456"
        assert claims["email"] == "test@test.com"

    def test_verify_jwt_expired_token(self):
        # Create a token that's already expired
        payload = {
            "user_id": "user-789",
            "email": "expired@test.com",
            "exp": datetime.datetime.utcnow() - datetime.timedelta(hours=1),
            "iat": datetime.datetime.utcnow() - datetime.timedelta(hours=2),
        }
        token = pyjwt.encode(payload, Config.JWT_SECRET, algorithm="HS256")
        with pytest.raises(pyjwt.ExpiredSignatureError):
            verify_jwt(token)

    def test_verify_jwt_tampered_token(self):
        token = create_jwt("user-000", "hacker@evil.com")
        # Tamper with the token
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(Exception):
            verify_jwt(tampered)

    def test_verify_jwt_missing_claims(self):
        """JWT with valid signature but missing user_id/email should raise ValueError."""
        payload = {
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1),
            "iat": datetime.datetime.utcnow(),
            # Missing user_id and email
        }
        token = pyjwt.encode(payload, Config.JWT_SECRET, algorithm="HS256")
        with pytest.raises(ValueError, match="missing required claims"):
            verify_jwt(token)

    def test_verify_jwt_empty_user_id(self):
        """JWT with empty user_id should raise ValueError."""
        payload = {
            "user_id": "",
            "email": "test@test.com",
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1),
            "iat": datetime.datetime.utcnow(),
        }
        token = pyjwt.encode(payload, Config.JWT_SECRET, algorithm="HS256")
        with pytest.raises(ValueError, match="missing required claims"):
            verify_jwt(token)


class TestVerifyPasswordEdgeCases:
    """Tests for verify_password handling of invalid hashes."""

    def test_invalid_bcrypt_hash_returns_false(self):
        """Invalid bcrypt hash should return False, not crash."""
        assert verify_password("password", "not_a_valid_hash") is False

    def test_empty_hash_returns_false(self):
        assert verify_password("password", "") is False
