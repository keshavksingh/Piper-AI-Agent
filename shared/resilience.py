"""Resilience patterns: retry, circuit breaker, timeout."""

import grpc
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

# Transient gRPC status codes that are safe to retry
_TRANSIENT_CODES = frozenset([
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.ABORTED,
])


def _is_transient_grpc_error(exception):
    """Return True only for transient gRPC errors worth retrying."""
    if not isinstance(exception, grpc.RpcError):
        return False
    try:
        return exception.code() in _TRANSIENT_CODES
    except (AttributeError, Exception):
        return False


# Retry decorator for gRPC calls — only retries transient errors
grpc_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception(_is_transient_grpc_error),
    reraise=True,
)


def create_grpc_channel(address: str):
    """Create a TLS-secured gRPC channel."""
    from shared.config import Config

    with open(Config.TLS_CA_CERT, "rb") as f:
        ca_cert = f.read()
    credentials = grpc.ssl_channel_credentials(root_certificates=ca_cert)
    options = [
        ("grpc.keepalive_time_ms", 30000),
        ("grpc.keepalive_timeout_ms", 10000),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),
    ]
    return grpc.secure_channel(address, credentials, options=options)
