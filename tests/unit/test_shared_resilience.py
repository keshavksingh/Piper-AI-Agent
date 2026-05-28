"""Tests for shared.resilience — Retry decorator and gRPC channel creation."""

from unittest.mock import patch, MagicMock, mock_open

import grpc
import pytest

from shared.resilience import grpc_retry, create_grpc_channel, _is_transient_grpc_error


def _make_transient_error():
    """Create a mock gRPC error with a transient status code (UNAVAILABLE)."""
    err = grpc.RpcError()
    err.code = lambda: grpc.StatusCode.UNAVAILABLE
    return err


def _make_non_transient_error():
    """Create a mock gRPC error with a non-transient status code (NOT_FOUND)."""
    err = grpc.RpcError()
    err.code = lambda: grpc.StatusCode.NOT_FOUND
    return err


class TestGrpcRetry:
    """Tests for the retry decorator."""

    def test_success_without_retry(self):
        call_count = {"n": 0}

        @grpc_retry
        def succeed():
            call_count["n"] += 1
            return "ok"

        result = succeed()
        assert result == "ok"
        assert call_count["n"] == 1

    def test_retry_on_transient_grpc_error(self):
        call_count = {"n": 0}

        @grpc_retry
        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise _make_transient_error()
            return "recovered"

        result = flaky()
        assert result == "recovered"
        assert call_count["n"] == 3

    def test_raises_after_max_attempts(self):
        @grpc_retry
        def always_fail():
            raise _make_transient_error()

        with pytest.raises(grpc.RpcError):
            always_fail()

    def test_no_retry_on_non_transient_error(self):
        """Non-transient errors (NOT_FOUND, PERMISSION_DENIED) should not be retried."""
        call_count = {"n": 0}

        @grpc_retry
        def not_found():
            call_count["n"] += 1
            raise _make_non_transient_error()

        with pytest.raises(grpc.RpcError):
            not_found()
        # Should only be called once — no retry
        assert call_count["n"] == 1


class TestIsTransientGrpcError:
    """Tests for the transient error predicate."""

    def test_unavailable_is_transient(self):
        err = grpc.RpcError()
        err.code = lambda: grpc.StatusCode.UNAVAILABLE
        assert _is_transient_grpc_error(err) is True

    def test_deadline_exceeded_is_transient(self):
        err = grpc.RpcError()
        err.code = lambda: grpc.StatusCode.DEADLINE_EXCEEDED
        assert _is_transient_grpc_error(err) is True

    def test_not_found_is_not_transient(self):
        err = grpc.RpcError()
        err.code = lambda: grpc.StatusCode.NOT_FOUND
        assert _is_transient_grpc_error(err) is False

    def test_non_grpc_error_is_false(self):
        assert _is_transient_grpc_error(ValueError("not grpc")) is False

    def test_missing_code_method_returns_false(self):
        """If exception.code() raises AttributeError, treat as non-transient."""
        err = grpc.RpcError()
        # Remove code method if present
        if hasattr(err, 'code'):
            del err.code
        assert _is_transient_grpc_error(err) is False


class TestCreateGrpcChannel:
    """Tests for TLS channel creation."""

    @patch("builtins.open", mock_open(read_data=b"fake-ca-cert"))
    @patch("grpc.ssl_channel_credentials")
    @patch("grpc.secure_channel")
    def test_creates_secure_channel(self, mock_secure_channel, mock_ssl_creds):
        mock_ssl_creds.return_value = MagicMock()
        mock_secure_channel.return_value = MagicMock()

        channel = create_grpc_channel("localhost:50051")

        mock_ssl_creds.assert_called_once_with(root_certificates=b"fake-ca-cert")
        mock_secure_channel.assert_called_once()
        # Verify address is passed
        args = mock_secure_channel.call_args
        assert args[0][0] == "localhost:50051"
