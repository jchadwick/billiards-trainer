"""Fail-closed, per-client WebSocket connection rate limiting."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import ipaddress
import logging
import os
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from urllib.parse import parse_qsl, urlsplit

from backend.config import config

logger = logging.getLogger(__name__)


class RateLimitUnavailable(RuntimeError):
    """Raised when the selected rate-limit store cannot enforce admission."""


class RateLimitStore(Protocol):
    """Storage contract shared by the local and Redis implementations."""

    async def allow(self, client_id: str, limit: int, window_seconds: int) -> bool:
        """Atomically record an attempt if it is within the configured limit."""

    async def ping(self) -> bool:
        """Return whether this store can currently enforce limits."""

    async def close(self) -> None:
        """Release resources owned by this store."""


class MemoryRateLimitStore:
    """Bounded per-process sliding-window store for a single API worker."""

    def __init__(
        self,
        max_clients: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_clients < 1:
            raise ValueError("max_clients must be positive")
        self.max_clients = max_clients
        self._clock = clock
        self._attempts: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def allow(self, client_id: str, limit: int, window_seconds: int) -> bool:
        if limit < 1 or window_seconds < 1:
            raise ValueError("Rate-limit values must be positive")

        async with self._lock:
            now = self._clock()
            self._expire_entries(now, window_seconds)
            attempts = self._attempts.get(client_id)
            if attempts is not None:
                self._discard_expired_attempts(attempts, now, window_seconds)
                if not attempts:
                    del self._attempts[client_id]
                    attempts = None

            if attempts is None:
                # Refuse new keys at capacity instead of evicting an active
                # client's history and silently weakening that client's limit.
                if len(self._attempts) >= self.max_clients:
                    return False
                attempts = deque()
                self._attempts[client_id] = attempts

            if len(attempts) >= limit:
                return False

            attempts.append(now)
            self._attempts.move_to_end(client_id)
            return True

    def _expire_entries(self, now: float, window_seconds: int) -> None:
        """Remove expired attempts and empty clients before admitting a key."""
        # Entries are ordered by their last admitted attempt. With one active
        # window per process, once the oldest client remains active, newer
        # clients cannot have expired histories.
        while self._attempts:
            client_id = next(iter(self._attempts))
            attempts = self._attempts[client_id]
            self._discard_expired_attempts(attempts, now, window_seconds)
            if attempts:
                break
            del self._attempts[client_id]

    @staticmethod
    def _discard_expired_attempts(
        attempts: deque[float], now: float, window_seconds: int
    ) -> None:
        while attempts and now - attempts[0] >= window_seconds:
            attempts.popleft()

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        async with self._lock:
            self._attempts.clear()

    @property
    def tracked_clients(self) -> int:
        """Number of client keys currently retained by the store."""
        return len(self._attempts)


_REDIS_ADMISSION_SCRIPT = """
local now = redis.call('TIME')
local now_ms = (tonumber(now[1]) * 1000) + math.floor(tonumber(now[2]) / 1000)
local window_ms = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local key = KEYS[1]
redis.call('ZREMRANGEBYSCORE', key, '-inf', now_ms - window_ms)
local count = redis.call('ZCARD', key)
if count >= limit then
    return 0
end
redis.call('ZADD', key, now_ms, ARGV[3])
redis.call('PEXPIRE', key, window_ms)
return 1
"""


class RedisRateLimitStore:
    """Shared sliding-window store using one atomic Redis Lua admission step.

    The optional store is anonymous like the rest of the application. Reject
    credentials embedded in URL userinfo or query parameters before the Redis
    client is imported or a connection is constructed.
    """

    def __init__(
        self,
        client: Any | None,
        initialization_error: str | None = None,
        timeout_seconds: float = 2.0,
    ):
        self._client = client
        self._initialization_error = initialization_error
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_url(
        cls, url: str | None, timeout_seconds: float = 2.0
    ) -> RedisRateLimitStore:
        if not url:
            return cls(
                None,
                "BILLIARDS_RATE_LIMIT_REDIS_URL is not configured",
                timeout_seconds,
            )
        try:
            parsed_url = urlsplit(url)
            auth_query_parameters = {
                "auth",
                "credential_provider",
                "pass",
                "password",
                "token",
                "user",
                "username",
            }
            has_auth_userinfo = (
                parsed_url.username is not None or parsed_url.password is not None
            )
            has_auth_query = any(
                key.lower() in auth_query_parameters
                for key, _ in parse_qsl(parsed_url.query, keep_blank_values=True)
            )
            if has_auth_userinfo or has_auth_query:
                return cls(
                    None,
                    "Redis URL must not contain authentication credentials",
                    timeout_seconds,
                )

            redis_asyncio = importlib.import_module("redis.asyncio")
            return cls(
                redis_asyncio.Redis.from_url(
                    url,
                    decode_responses=True,
                    socket_connect_timeout=timeout_seconds,
                    socket_timeout=timeout_seconds,
                ),
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            return cls(
                None,
                f"Redis client could not be initialized: {exc}",
                timeout_seconds,
            )

    async def allow(self, client_id: str, limit: int, window_seconds: int) -> bool:
        if limit < 1 or window_seconds < 1:
            raise ValueError("Rate-limit values must be positive")
        if self._client is None:
            raise RuntimeError(self._initialization_error or "Redis is unavailable")

        # Hash client addresses so Redis keys do not expose network identifiers.
        digest = hashlib.sha256(client_id.encode("utf-8")).hexdigest()
        key = f"billiards-trainer:ws-rate-limit:{digest}"
        result = await asyncio.wait_for(
            self._client.eval(
                _REDIS_ADMISSION_SCRIPT,
                1,
                key,
                window_seconds * 1000,
                limit,
                uuid.uuid4().hex,
            ),
            timeout=self._timeout_seconds,
        )
        return bool(result)

    async def ping(self) -> bool:
        if self._client is None:
            raise RuntimeError(self._initialization_error or "Redis is unavailable")
        await asyncio.wait_for(self._client.ping(), timeout=self._timeout_seconds)
        return True

    async def close(self) -> None:
        if self._client is None:
            return
        close = getattr(self._client, "aclose", None) or getattr(
            self._client, "close", None
        )
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await asyncio.wait_for(result, timeout=self._timeout_seconds)


@dataclass(frozen=True)
class RateLimitSettings:
    """Validated runtime settings for WebSocket connection admission."""

    store: str
    limit: int
    window_seconds: int
    trusted_proxies: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]
    max_tracked_ips: int
    store_timeout_seconds: float

    @classmethod
    def from_config(cls, config_get: Callable[[str, Any], Any]) -> RateLimitSettings:
        store_name = config_get("api.rate_limiting.store", "memory")
        if store_name not in {"memory", "redis"}:
            raise ValueError("api.rate_limiting.store must be 'memory' or 'redis'")

        limit = _positive_int(
            config_get("api.rate_limiting.websocket_connections_per_ip", 10),
            "api.rate_limiting.websocket_connections_per_ip",
        )
        window_seconds = _positive_int(
            config_get("api.rate_limiting.window_seconds", 60),
            "api.rate_limiting.window_seconds",
        )
        max_tracked_ips = _positive_int(
            config_get("api.rate_limiting.max_tracked_ips", 10000),
            "api.rate_limiting.max_tracked_ips",
        )
        store_timeout_seconds = _positive_finite_number(
            config_get("api.rate_limiting.store_timeout_seconds", 2.0),
            "api.rate_limiting.store_timeout_seconds",
        )
        proxy_values = config_get("api.rate_limiting.trusted_proxies", [])
        if not isinstance(proxy_values, list):
            raise ValueError("api.rate_limiting.trusted_proxies must be a list")
        try:
            trusted_proxies = tuple(
                ipaddress.ip_network(value, strict=False) for value in proxy_values
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Trusted proxy entries must be valid IP CIDRs") from exc

        return cls(
            store=store_name,
            limit=limit,
            window_seconds=window_seconds,
            trusted_proxies=trusted_proxies,
            max_tracked_ips=max_tracked_ips,
            store_timeout_seconds=store_timeout_seconds,
        )


def _positive_int(value: Any, setting: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{setting} must be a positive integer")
    return int(value)


def _positive_finite_number(value: Any, setting: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{setting} must be a positive finite number")
    number = float(value)
    if not (0 < number < float("inf")):
        raise ValueError(f"{setting} must be a positive finite number")
    return number


class ConnectionRateLimiter:
    """Manage configured store lifecycle and fail-closed admission checks."""

    def __init__(
        self,
        config_get: Callable[[str, Any], Any] = config.get,
        environ: Any = os.environ,
    ) -> None:
        self._config_get = config_get
        self._environ = environ
        self.settings = RateLimitSettings.from_config(config_get)
        self.store: RateLimitStore = MemoryRateLimitStore(self.settings.max_tracked_ips)
        self._ready = True

    async def initialize(self) -> bool:
        """Select the configured store and check dependency health."""
        try:
            settings = RateLimitSettings.from_config(self._config_get)
            await asyncio.wait_for(
                self.store.close(), timeout=settings.store_timeout_seconds
            )
            self.settings = settings
            if self.settings.store == "memory":
                self.store = MemoryRateLimitStore(self.settings.max_tracked_ips)
            else:
                self.store = RedisRateLimitStore.from_url(
                    self._environ.get("BILLIARDS_RATE_LIMIT_REDIS_URL"),
                    timeout_seconds=self.settings.store_timeout_seconds,
                )
            self._ready = await asyncio.wait_for(
                self.store.ping(), timeout=self.settings.store_timeout_seconds
            )
        except Exception as exc:
            # Keep malformed configuration and client initialization failures
            # closed even if the previous store was a healthy memory instance.
            self.store = RedisRateLimitStore(
                None, f"Rate-limit store initialization failed: {exc}"
            )
            self._ready = False
            logger.exception("WebSocket rate-limit store initialization failed")
        if not self._ready:
            logger.error(
                "WebSocket rate-limit store is unavailable; admission fails closed"
            )
        return self._ready

    async def check_rate_limit(self, client_ip: str) -> bool:
        """Consume one connection attempt or fail closed on store failure."""
        try:
            allowed = await asyncio.wait_for(
                self.store.allow(
                    client_ip, self.settings.limit, self.settings.window_seconds
                ),
                timeout=self.settings.store_timeout_seconds,
            )
            self._ready = True
            return allowed
        except Exception as exc:
            self._ready = False
            raise RateLimitUnavailable("Rate-limit store is unavailable") from exc

    async def check_readiness(self) -> bool:
        """Probe the selected store so readiness reflects current dependency state."""
        try:
            self._ready = await asyncio.wait_for(
                self.store.ping(), timeout=self.settings.store_timeout_seconds
            )
        except Exception:
            self._ready = False
            logger.warning("WebSocket rate-limit store readiness check failed")
        return self._ready

    async def close(self) -> None:
        await asyncio.wait_for(
            self.store.close(), timeout=self.settings.store_timeout_seconds
        )
        if self.settings.store == "redis":
            self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready


rate_limiter = ConnectionRateLimiter()


def resolve_client_ip(
    peer_host: str | None,
    headers: Any,
    trusted_proxies: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...],
) -> str:
    """Resolve the remote client while accepting proxy headers only from trusted hops."""
    if not peer_host:
        return "unknown"
    try:
        peer = ipaddress.ip_address(peer_host)
    except ValueError:
        return peer_host

    if not _is_trusted(peer, trusted_proxies):
        return str(peer)

    forwarded_for = headers.get("x-forwarded-for")
    if forwarded_for:
        try:
            chain = [
                ipaddress.ip_address(item.strip()) for item in forwarded_for.split(",")
            ]
        except ValueError:
            return str(peer)

        client = peer
        for hop in reversed(chain):
            if not _is_trusted(client, trusted_proxies):
                break
            client = hop
        return str(client)

    real_ip = headers.get("x-real-ip")
    if real_ip:
        try:
            return str(ipaddress.ip_address(real_ip.strip()))
        except ValueError:
            return str(peer)
    return str(peer)


def _is_trusted(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...],
) -> bool:
    return any(
        address.version == network.version and address in network
        for network in networks
    )
