"""Unit tests for WebSocket connection admission stores and proxy resolution."""

import asyncio
import ipaddress
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, WebSocketDisconnect
from starlette.routing import Match

from backend.api import main as api_main
from backend.api.routes import health
from backend.api.websocket import endpoints
from backend.api.websocket.monitoring import QualityMetrics
from backend.api.websocket.rate_limit import (
    ConnectionRateLimiter,
    MemoryRateLimitStore,
    RateLimitSettings,
    RateLimitUnavailable,
    RedisRateLimitStore,
    resolve_client_ip,
)
from backend.tests.support.compat_api import create_compatibility_app


class FakeRedis:
    """Small deterministic Redis stand-in shared by separate store instances."""

    def __init__(self):
        self.now_ms = 1_000_000
        self.entries = {}
        self.lock = asyncio.Lock()
        self.available = True

    async def ping(self):
        if not self.available:
            raise ConnectionError("Redis unavailable")
        return True

    async def eval(self, script, numkeys, *args):
        assert "redis.call('TIME')" in script
        assert "redis.call('ZADD'" in script
        assert numkeys == 1
        key, window_ms, limit, token = args
        async with self.lock:
            if not self.available:
                raise ConnectionError("Redis unavailable")
            current = self.entries.setdefault(key, {})
            cutoff = self.now_ms - int(window_ms)
            current = {
                entry: timestamp
                for entry, timestamp in current.items()
                if timestamp > cutoff
            }
            self.entries[key] = current
            if len(current) >= int(limit):
                return 0
            current[token] = self.now_ms
            return 1

    async def aclose(self):
        return None


def settings(**overrides):
    values = {
        "api.rate_limiting.store": "memory",
        "api.rate_limiting.websocket_connections_per_ip": 2,
        "api.rate_limiting.window_seconds": 60,
        "api.rate_limiting.trusted_proxies": [],
        "api.rate_limiting.max_tracked_ips": 8,
        "api.rate_limiting.store_timeout_seconds": 2.0,
    }
    values.update(overrides)
    return lambda key, default=None: values.get(key, default)


class _DisconnectingSocket:
    client = SimpleNamespace(host="203.0.113.9")
    headers = {"x-request-id": "attacker-supplied", "user-agent": "test"}

    async def receive_text(self):
        raise WebSocketDisconnect()

    async def close(self, code=None, reason=None):
        return None


@pytest.mark.asyncio
async def test_memory_store_expires_attempts_and_reclaims_client_keys():
    current_time = [0.0]
    store = MemoryRateLimitStore(max_clients=1, clock=lambda: current_time[0])

    assert await store.allow("192.0.2.1", limit=1, window_seconds=10)
    assert not await store.allow("192.0.2.1", limit=1, window_seconds=10)
    assert not await store.allow("192.0.2.2", limit=1, window_seconds=10)

    current_time[0] = 10.0
    assert await store.allow("192.0.2.2", limit=1, window_seconds=10)
    assert store.tracked_clients == 1


@pytest.mark.asyncio
async def test_shared_redis_instances_enforce_one_atomic_quota_and_expire():
    redis = FakeRedis()
    first = RedisRateLimitStore(redis)
    second = RedisRateLimitStore(redis)

    assert await first.allow("198.51.100.9", limit=2, window_seconds=60)
    assert await second.allow("198.51.100.9", limit=2, window_seconds=60)
    assert not await first.allow("198.51.100.9", limit=2, window_seconds=60)

    redis.now_ms += 60_001
    assert await second.allow("198.51.100.9", limit=2, window_seconds=60)


@pytest.mark.asyncio
async def test_explicit_redis_without_url_fails_closed_and_is_unready():
    limiter = ConnectionRateLimiter(
        config_get=settings(**{"api.rate_limiting.store": "redis"}), environ={}
    )

    assert not await limiter.initialize()
    assert not limiter.ready
    with pytest.raises(RateLimitUnavailable):
        await limiter.check_rate_limit("198.51.100.9")
    assert not await limiter.check_readiness()


@pytest.mark.asyncio
async def test_memory_mode_does_not_import_the_optional_redis_client(monkeypatch):
    import backend.api.websocket.rate_limit as rate_limit_module

    def reject_redis_import(module_name):
        if module_name == "redis.asyncio":
            raise AssertionError("Memory mode must not import Redis")
        raise AssertionError(f"Unexpected dynamic import: {module_name}")

    monkeypatch.setattr(
        rate_limit_module.importlib, "import_module", reject_redis_import
    )
    limiter = ConnectionRateLimiter(config_get=settings(), environ={})

    assert await limiter.initialize()
    assert await limiter.check_rate_limit("198.51.100.9")


@pytest.mark.asyncio
async def test_readiness_reports_redis_store_unavailable(monkeypatch):
    redis = FakeRedis()
    redis.available = False
    monkeypatch.setattr(
        RedisRateLimitStore,
        "from_url",
        classmethod(
            lambda cls, url, timeout_seconds=2.0: RedisRateLimitStore(
                redis, timeout_seconds=timeout_seconds
            )
        ),
    )
    limiter = ConnectionRateLimiter(
        config_get=settings(**{"api.rate_limiting.store": "redis"}),
        environ={"BILLIARDS_RATE_LIMIT_REDIS_URL": "redis://configured"},
    )
    assert not await limiter.initialize()
    monkeypatch.setattr(health, "rate_limiter", limiter)
    app_state = SimpleNamespace(
        core_module=object(), config_module=object(), websocket_manager=object()
    )

    with pytest.raises(HTTPException) as error:
        await health.readiness_check(app_state)

    assert error.value.status_code == 503
    assert error.value.detail["components"]["rate_limit_store"] == "unavailable"


@pytest.mark.asyncio
async def test_endpoint_generates_and_passes_its_own_request_id(monkeypatch):
    handler = SimpleNamespace(
        connect=AsyncMock(return_value="client-id"),
        disconnect=AsyncMock(),
        handle_message=AsyncMock(),
    )
    limiter = SimpleNamespace(check_rate_limit=AsyncMock(return_value=True))
    monitor = endpoints.connection_monitor
    monitor.client_metrics.pop("client-id", None)
    monkeypatch.setattr(endpoints, "websocket_handler", handler)
    monkeypatch.setattr(endpoints, "rate_limiter", limiter)
    monkeypatch.setattr(
        endpoints.subscription_manager, "remove_all_subscriptions", AsyncMock()
    )

    try:
        await endpoints.websocket_endpoint(
            _DisconnectingSocket(), client_ip="203.0.113.9"
        )

        metadata = handler.connect.await_args.kwargs["metadata"]
        assert metadata["request_id"] != "attacker-supplied"
        assert len(metadata["request_id"]) == 36
        assert isinstance(monitor.client_metrics["client-id"], QualityMetrics)
        assert monitor.client_metrics["client-id"].client_id == "client-id"
    finally:
        monitor.client_metrics.pop("client-id", None)


def test_application_registers_only_the_canonical_websocket_route():
    app = create_compatibility_app()

    def has_websocket_route(path: str) -> bool:
        scope = {
            "type": "websocket",
            "path": path,
            "root_path": "",
            "app": app,
            "endpoint": None,
            "router": app.router,
        }
        return any(
            route.matches(scope)[0] is Match.FULL
            for route in app.routes
            if hasattr(route, "matches")
        )

    assert has_websocket_route("/api/v1/websocket/ws")
    assert not has_websocket_route("/api/v1/ws")


@pytest.mark.asyncio
async def test_redis_outage_marks_readiness_unhealthy_and_rejects_admission(
    monkeypatch,
):
    redis = FakeRedis()
    redis.available = False
    monkeypatch.setattr(
        RedisRateLimitStore,
        "from_url",
        classmethod(
            lambda cls, url, timeout_seconds=2.0: RedisRateLimitStore(
                redis, timeout_seconds=timeout_seconds
            )
        ),
    )
    limiter = ConnectionRateLimiter(
        config_get=settings(**{"api.rate_limiting.store": "redis"}),
        environ={"BILLIARDS_RATE_LIMIT_REDIS_URL": "redis://configured"},
    )

    assert not await limiter.initialize()
    with pytest.raises(RateLimitUnavailable):
        await limiter.check_rate_limit("198.51.100.9")
    assert not await limiter.check_readiness()


@pytest.mark.asyncio
async def test_nonresponsive_redis_ping_times_out_during_initialization(monkeypatch):
    class GatedPingRedis:
        async def ping(self):
            await asyncio.Event().wait()

        async def aclose(self):
            return None

    redis = GatedPingRedis()
    monkeypatch.setattr(
        RedisRateLimitStore,
        "from_url",
        classmethod(
            lambda cls, url, timeout_seconds=2.0: RedisRateLimitStore(
                redis, timeout_seconds=timeout_seconds
            )
        ),
    )
    limiter = ConnectionRateLimiter(
        config_get=settings(
            **{
                "api.rate_limiting.store": "redis",
                "api.rate_limiting.store_timeout_seconds": 0.01,
            }
        ),
        environ={"BILLIARDS_RATE_LIMIT_REDIS_URL": "redis://configured"},
    )

    assert not await limiter.initialize()
    assert not limiter.ready
    assert not await limiter.check_readiness()


@pytest.mark.asyncio
async def test_nonresponsive_redis_readiness_probe_times_out(monkeypatch):
    class GatedPingRedis:
        def __init__(self):
            self.gated = False
            self.release = asyncio.Event()

        async def ping(self):
            if self.gated:
                await self.release.wait()
            return True

        async def aclose(self):
            return None

    redis = GatedPingRedis()
    monkeypatch.setattr(
        RedisRateLimitStore,
        "from_url",
        classmethod(
            lambda cls, url, timeout_seconds=2.0: RedisRateLimitStore(
                redis, timeout_seconds=timeout_seconds
            )
        ),
    )
    limiter = ConnectionRateLimiter(
        config_get=settings(
            **{
                "api.rate_limiting.store": "redis",
                "api.rate_limiting.store_timeout_seconds": 0.01,
            }
        ),
        environ={"BILLIARDS_RATE_LIMIT_REDIS_URL": "redis://configured"},
    )
    assert await limiter.initialize()
    redis.gated = True

    assert not await limiter.check_readiness()
    assert not limiter.ready


@pytest.mark.asyncio
async def test_nonresponsive_redis_admission_times_out_and_fails_closed(monkeypatch):
    class GatedAdmissionRedis:
        async def ping(self):
            return True

        async def eval(self, *_args):
            await asyncio.Event().wait()

        async def aclose(self):
            return None

    redis = GatedAdmissionRedis()
    monkeypatch.setattr(
        RedisRateLimitStore,
        "from_url",
        classmethod(
            lambda cls, url, timeout_seconds=2.0: RedisRateLimitStore(
                redis, timeout_seconds=timeout_seconds
            )
        ),
    )
    limiter = ConnectionRateLimiter(
        config_get=settings(
            **{
                "api.rate_limiting.store": "redis",
                "api.rate_limiting.store_timeout_seconds": 0.01,
            }
        ),
        environ={"BILLIARDS_RATE_LIMIT_REDIS_URL": "redis://configured"},
    )
    assert await limiter.initialize()

    with pytest.raises(RateLimitUnavailable):
        await limiter.check_rate_limit("198.51.100.9")
    assert not limiter.ready


def test_redis_timeout_configuration_must_be_positive_and_finite():
    for invalid in (True, 0, -1, float("inf"), float("nan"), "2"):
        with pytest.raises(ValueError):
            RateLimitSettings.from_config(
                settings(**{"api.rate_limiting.store_timeout_seconds": invalid})
            )


def test_redis_client_receives_connect_and_read_deadlines(monkeypatch):
    import backend.api.websocket.rate_limit as rate_limit_module

    captured = {}

    class FakeRedisClientFactory:
        @staticmethod
        def from_url(url, **options):
            captured["url"] = url
            captured.update(options)
            return FakeRedis()

    monkeypatch.setattr(
        rate_limit_module.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(Redis=FakeRedisClientFactory),
    )

    RedisRateLimitStore.from_url("redis://configured", timeout_seconds=0.25)

    assert captured["socket_connect_timeout"] == 0.25
    assert captured["socket_timeout"] == 0.25


@pytest.mark.parametrize(
    "url",
    [
        "redis://user:password@redis:6379/0",
        "redis://:password@redis:6379/0",
        "redis://redis:6379/0?username=user",
        "redis://redis:6379/0?password=password",
        "redis://redis:6379/0?credential_provider=provider",
    ],
)
def test_redis_store_rejects_authentication_parameters_before_client_import(
    monkeypatch, url
):
    import backend.api.websocket.rate_limit as rate_limit_module

    def fail_if_imported(module_name):
        raise AssertionError(f"Redis client must not load for {module_name}")

    monkeypatch.setattr(rate_limit_module.importlib, "import_module", fail_if_imported)

    store = RedisRateLimitStore.from_url(url)

    assert store._client is None
    assert store._initialization_error == (
        "Redis URL must not contain authentication credentials"
    )


@pytest.mark.asyncio
async def test_authenticated_redis_url_keeps_rate_limiter_fail_closed():
    limiter = ConnectionRateLimiter(
        config_get=settings(**{"api.rate_limiting.store": "redis"}),
        environ={
            "BILLIARDS_RATE_LIMIT_REDIS_URL": "redis://user:password@redis:6379/0"
        },
    )

    assert not await limiter.initialize()
    assert not limiter.ready
    with pytest.raises(RateLimitUnavailable):
        await limiter.check_rate_limit("198.51.100.9")


def test_forwarded_headers_from_untrusted_peer_cannot_spoof_client_address():
    resolved = resolve_client_ip(
        "203.0.113.7",
        {"x-forwarded-for": "198.51.100.2", "x-real-ip": "198.51.100.3"},
        (),
    )

    assert resolved == "203.0.113.7"


def test_trusted_proxy_chain_walks_back_to_first_untrusted_hop():
    resolved = resolve_client_ip(
        "10.0.0.2",
        {"x-forwarded-for": "198.51.100.5, 10.0.0.1"},
        (ipaddress.ip_network("10.0.0.0/8"),),
    )

    assert resolved == "198.51.100.5"


def test_malformed_forwarded_header_falls_back_to_proxy_peer():
    resolved = resolve_client_ip(
        "10.0.0.2",
        {"x-forwarded-for": "not-an-ip"},
        (ipaddress.ip_network("10.0.0.0/8"),),
    )

    assert resolved == "10.0.0.2"
