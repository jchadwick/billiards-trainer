"""Deterministic, hardware-free contracts for graceful shutdown coordination."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

import backend.api.shutdown as shutdown

if TYPE_CHECKING:
    from collections.abc import Callable


def _config(**overrides: Any) -> shutdown.ShutdownConfig:
    """Use bounded, fast timeouts and avoid real persistence or active-operation waits."""
    settings: dict[str, Any] = {
        "graceful_timeout": 2.0,
        "module_timeout": 1.0,
        "connection_drain_timeout": 1.0,
        "background_task_timeout": 1.0,
        "save_state_on_shutdown": False,
        "force_kill_on_timeout": True,
        "wait_for_active_operations": False,
        "max_module_retries": 0,
        "retry_delay": 0.0,
    }
    settings.update(overrides)
    return shutdown.ShutdownConfig(**settings)


@pytest.mark.unit
def test_initial_status_reports_not_started_state() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config())

    assert coordinator.get_shutdown_status() == {
        "is_shutting_down": False,
        "phase": "initiated",
        "status": "not_started",
        "started_at": None,
        "completed_at": None,
        "elapsed_time": 0.0,
        "modules_completed": [],
        "modules_failed": [],
        "active_operations": 0,
        "errors": [],
        "warnings": [],
        "force_requested": False,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_runs_phases_callbacks_and_modules_in_dependency_order() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config(save_state_on_shutdown=True))
    events: list[str] = []

    def callback() -> None:
        events.append("stop-requests")

    coordinator.add_shutdown_callback(callback)

    async def mark(name: str) -> None:
        events.append(name)

    coordinator._drain_connections = lambda: mark("drain")  # type: ignore[method-assign]
    coordinator._stop_background_tasks = lambda: mark("background")  # type: ignore[method-assign]
    coordinator._save_system_state = lambda: mark("save")  # type: ignore[method-assign]
    coordinator._cleanup_resources = lambda: mark("cleanup")  # type: ignore[method-assign]
    coordinator._finalize_shutdown = lambda: mark("finalize")  # type: ignore[method-assign]

    def register_sync(name: str) -> Callable[[], None]:
        def shutdown_sync() -> None:
            events.append(name)

        return shutdown_sync

    async def shutdown_vision() -> None:
        events.append("vision")

    coordinator.register_module_shutdown("custom", register_sync("custom"))
    coordinator.register_module_shutdown("config", register_sync("config"))
    coordinator.register_module_shutdown("vision", shutdown_vision)
    coordinator.register_module_shutdown("core", register_sync("core"))
    coordinator.register_module_shutdown("websocket", register_sync("websocket"))

    assert await coordinator.initiate_shutdown(save_state=True) is True
    assert events == [
        "stop-requests",
        "drain",
        "background",
        "save",
        "cleanup",
        "vision",
        "core",
        "websocket",
        "config",
        "custom",
        "finalize",
    ]
    status = coordinator.get_shutdown_status()
    assert status["phase"] == "completed"
    assert status["status"] == "completed"
    assert status["is_shutting_down"] is True
    assert status["started_at"] is not None
    assert status["completed_at"] is not None
    assert status["modules_completed"] == [
        "vision",
        "core",
        "websocket",
        "config",
        "custom",
    ]
    assert status["modules_failed"] == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_omits_state_save_when_call_or_config_disables_it() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config(save_state_on_shutdown=False))
    save_calls = 0

    async def save() -> None:
        nonlocal save_calls
        save_calls += 1

    coordinator._save_system_state = save  # type: ignore[method-assign]

    assert await coordinator.initiate_shutdown(save_state=True)
    assert save_calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_waits_for_active_operations_only_when_configured_and_not_forced() -> (
    None
):
    coordinator = shutdown.ShutdownCoordinator(_config(wait_for_active_operations=True))
    waits = 0

    async def wait_for_operations() -> None:
        nonlocal waits
        waits += 1

    coordinator._wait_for_active_operations = wait_for_operations  # type: ignore[method-assign]

    assert await coordinator._execute_shutdown_sequence(save_state=False)
    assert waits == 1

    coordinator._force_shutdown = True
    assert await coordinator._execute_shutdown_sequence(save_state=False)
    assert waits == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sequence_reports_elapsed_shutdown_over_graceful_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = shutdown.ShutdownCoordinator(_config(graceful_timeout=0.1))
    times = iter([0.0, 1.0])

    async def no_op() -> None:
        return None

    coordinator._stop_new_requests = no_op  # type: ignore[method-assign]
    coordinator._drain_connections = no_op  # type: ignore[method-assign]
    coordinator._stop_background_tasks = no_op  # type: ignore[method-assign]
    coordinator._cleanup_resources = no_op  # type: ignore[method-assign]
    coordinator._shutdown_modules = no_op  # type: ignore[method-assign]
    coordinator._finalize_shutdown = no_op  # type: ignore[method-assign]
    monkeypatch.setattr(shutdown, "time", SimpleNamespace(time=lambda: next(times)))

    assert await coordinator._execute_shutdown_sequence(save_state=False)
    assert len(coordinator.progress.warnings) == 1
    assert coordinator.progress.warnings[0].startswith("Shutdown took 1.00s")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_completed_shutdown_is_idempotent_and_does_not_rerun_callbacks() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config())
    callback_calls = 0

    def callback() -> None:
        nonlocal callback_calls
        callback_calls += 1

    coordinator.add_shutdown_callback(callback)

    assert await coordinator.initiate_shutdown(save_state=False)
    assert await coordinator.initiate_shutdown(save_state=False)
    assert callback_calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_callback_errors_are_warnings_and_do_not_stop_shutdown() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config())
    called: list[str] = []

    def fail() -> None:
        raise RuntimeError("listener refused")

    coordinator.add_shutdown_callback(fail)
    coordinator.add_shutdown_callback(lambda: called.append("continued"))

    assert await coordinator.initiate_shutdown(save_state=False)
    assert called == ["continued"]
    assert coordinator.progress.warnings == [
        "Shutdown callback error: listener refused"
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_connection_drain_timeout_is_warning_and_shutdown_continues() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config(connection_drain_timeout=0.001))

    async def never_drains() -> None:
        await asyncio.Event().wait()

    coordinator._wait_for_websocket_connections = never_drains  # type: ignore[method-assign]

    await coordinator._drain_connections()

    assert coordinator.progress.warnings == ["Connection drain timed out"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_background_tasks_are_cancelled_and_removed_when_finished() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config())
    started = asyncio.Event()

    async def worker() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(worker())
    coordinator.register_background_task(task)
    await started.wait()
    assert task in coordinator._background_tasks

    await coordinator._stop_background_tasks()
    await asyncio.sleep(0)

    assert task.cancelled()
    assert coordinator._background_tasks == set()
    assert coordinator.progress.warnings == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_background_task_registry_skips_tasks_that_already_finished() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config())
    completed = asyncio.create_task(asyncio.sleep(0))
    await completed
    coordinator._background_tasks.add(completed)

    await coordinator._stop_background_tasks()

    assert completed.done()
    assert not completed.cancelled()
    assert coordinator._background_tasks == set()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_background_task_timeout_is_recorded_and_registry_is_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = shutdown.ShutdownCoordinator(_config())
    task = asyncio.create_task(asyncio.sleep(60))
    coordinator.register_background_task(task)

    async def timeout(_awaitable: Any, *, timeout: float) -> Any:
        del timeout
        raise asyncio.TimeoutError

    async_namespace = SimpleNamespace(
        wait_for=timeout,
        gather=lambda *_args, **_kwargs: _CloseableAwaitable(),
        TimeoutError=asyncio.TimeoutError,
    )
    monkeypatch.setattr(shutdown, "asyncio", async_namespace)

    await coordinator._stop_background_tasks()
    await asyncio.gather(task, return_exceptions=True)

    assert task.cancelled()
    assert coordinator._background_tasks == set()
    assert coordinator.progress.warnings == ["Background task cleanup timed out"]


class _CloseableAwaitable:
    """Provide a harmless awaitable for the injected wait-for timeout path."""

    def __await__(self) -> Any:
        async def never() -> None:
            await asyncio.Event().wait()

        return never().__await__()

    def close(self) -> None:
        """Match coroutine cleanup performed by wait_for when it is cancelled."""


@pytest.mark.unit
@pytest.mark.asyncio
async def test_active_operation_wait_reports_operations_that_remain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = shutdown.ShutdownCoordinator(_config(graceful_timeout=0.01))
    coordinator.register_active_operation("frame-42")
    times = iter([0.0, 0.0, 1.0])

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(shutdown, "time", SimpleNamespace(time=lambda: next(times)))
    monkeypatch.setattr(shutdown, "asyncio", SimpleNamespace(sleep=no_wait))

    await coordinator._wait_for_active_operations()

    assert coordinator.progress.warnings == [
        "Active operations did not complete: ['frame-42']"
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_active_operation_wait_returns_when_work_finishes_or_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = shutdown.ShutdownCoordinator(_config(graceful_timeout=0.01))
    await coordinator._wait_for_active_operations()
    assert coordinator.progress.warnings == []

    coordinator.register_active_operation("frame-43")
    times = iter([0.0, 0.0])

    async def complete_operation(_seconds: float) -> None:
        coordinator.unregister_active_operation("frame-43")

    monkeypatch.setattr(shutdown, "time", SimpleNamespace(time=lambda: next(times)))
    monkeypatch.setattr(shutdown, "asyncio", SimpleNamespace(sleep=complete_operation))

    await coordinator._wait_for_active_operations()

    assert coordinator._active_operations == set()
    assert coordinator.progress.warnings == []


@pytest.mark.unit
def test_active_operation_registration_and_removal_keep_count_consistent() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config())

    coordinator.register_active_operation("frame-1")
    coordinator.register_active_operation("frame-2")
    coordinator.register_active_operation("frame-2")
    assert coordinator.progress.active_operations == 2

    coordinator.unregister_active_operation("frame-1")
    coordinator.unregister_active_operation("missing")
    assert coordinator.progress.active_operations == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_state_save_and_resource_cleanup_errors_are_captured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = shutdown.ShutdownCoordinator(_config(save_state_on_shutdown=True))

    async def fail_sleep(_seconds: float) -> None:
        raise OSError("resource unavailable")

    monkeypatch.setattr(shutdown, "asyncio", SimpleNamespace(sleep=fail_sleep))

    await coordinator._save_system_state()
    await coordinator._cleanup_resources()
    assert coordinator.progress.errors == [
        "State save failed: resource unavailable",
        "Resource cleanup failed: resource unavailable",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_state_save_and_resource_cleanup_complete_without_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = shutdown.ShutdownCoordinator(_config())

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(shutdown, "asyncio", SimpleNamespace(sleep=no_wait))

    await coordinator._save_system_state()
    await coordinator._cleanup_resources()

    assert coordinator.progress.errors == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_module_retries_coroutine_exception_then_succeeds() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config(max_module_retries=1))
    calls = 0

    async def flaky_module() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient failure")

    coordinator.register_module_shutdown("vision", flaky_module)

    assert await coordinator._shutdown_single_module("vision")
    assert calls == 2
    assert coordinator.progress.modules_completed == ["vision"]
    assert coordinator.progress.modules_failed == []
    assert coordinator.progress.errors == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_module_timeout_retries_then_records_failure() -> None:
    coordinator = shutdown.ShutdownCoordinator(
        _config(module_timeout=0.001, max_module_retries=1)
    )
    calls = 0

    async def stuck_module() -> None:
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    coordinator.register_module_shutdown("vision", stuck_module)

    assert await coordinator._shutdown_single_module("vision") is False
    assert calls == 2
    assert coordinator.progress.modules_completed == []
    assert coordinator.progress.modules_failed == ["vision"]
    assert coordinator.progress.errors == [
        "Module vision shutdown timed out (attempt 2)"
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sync_module_failure_is_retried_and_reported() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config(max_module_retries=1))
    calls = 0

    def broken_module() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("release failed")

    coordinator.register_module_shutdown("camera", broken_module)

    assert await coordinator._shutdown_single_module("camera") is False
    assert calls == 2
    assert coordinator.progress.modules_failed == ["camera"]
    assert coordinator.progress.errors == [
        "Module camera shutdown failed: release failed (attempt 2)"
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_module_shutdown_is_a_no_op_success() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config())

    assert await coordinator._shutdown_single_module("not-registered")
    assert coordinator.progress.modules_completed == []
    assert coordinator.progress.modules_failed == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_timeout_records_failure_and_forces_cleanup() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config(force_kill_on_timeout=True))
    task = asyncio.create_task(asyncio.sleep(60))
    coordinator.register_background_task(task)
    coordinator.register_active_operation("active-frame")

    async def raise_timeout() -> None:
        raise asyncio.TimeoutError

    coordinator._stop_new_requests = raise_timeout  # type: ignore[method-assign]

    assert await coordinator._execute_shutdown_sequence(save_state=False) is False
    await asyncio.gather(task, return_exceptions=True)

    assert task.cancelled()
    assert coordinator._active_operations == set()
    assert coordinator._background_tasks == set()
    assert coordinator.progress.status is shutdown.ShutdownStatus.FORCED
    assert coordinator.progress.errors == ["Shutdown sequence timed out"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_timeout_without_force_keeps_registered_work() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config(force_kill_on_timeout=False))
    coordinator.register_active_operation("active-frame")

    async def raise_timeout() -> None:
        raise asyncio.TimeoutError

    coordinator._stop_new_requests = raise_timeout  # type: ignore[method-assign]

    assert await coordinator._execute_shutdown_sequence(save_state=False) is False
    assert coordinator._active_operations == {"active-frame"}
    assert coordinator.progress.errors == ["Shutdown sequence timed out"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unexpected_sequence_exception_is_recorded_without_escaping() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config())

    async def fail() -> None:
        raise RuntimeError("callback subsystem unavailable")

    coordinator._stop_new_requests = fail  # type: ignore[method-assign]

    assert await coordinator._execute_shutdown_sequence(save_state=False) is False
    assert coordinator.progress.errors == [
        "Shutdown sequence error: callback subsystem unavailable"
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_initiate_shutdown_catches_unexpected_outer_exception() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config())

    async def fail(save_state: bool) -> bool:
        del save_state
        raise RuntimeError("sequence crashed")

    coordinator._execute_shutdown_sequence = fail  # type: ignore[method-assign]

    assert await coordinator.initiate_shutdown(save_state=False) is False
    assert coordinator.progress.status is shutdown.ShutdownStatus.FAILED
    assert coordinator.progress.phase is shutdown.ShutdownPhase.FAILED
    assert coordinator.progress.completed_at is None
    assert coordinator.progress.errors == ["Shutdown error: sequence crashed"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_public_helpers_use_registered_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = shutdown.ShutdownCoordinator(_config())
    monkeypatch.setattr(shutdown, "shutdown_coordinator", coordinator)

    async def stop_module() -> None:
        return None

    await shutdown.register_module_for_shutdown("vision", stop_module)
    assert await shutdown.initiate_system_shutdown(force=True, save_state=False)
    assert coordinator.progress.force_requested is True
    assert shutdown.get_shutdown_progress() == coordinator.get_shutdown_status()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_forced_shutdown_cancels_background_tasks_and_clears_operations() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config())
    task = asyncio.create_task(asyncio.sleep(60))
    completed = asyncio.create_task(asyncio.sleep(0))
    await completed
    coordinator.register_background_task(task)
    coordinator._background_tasks.add(completed)
    coordinator.register_active_operation("frame-1")

    await coordinator._force_shutdown_all()
    await asyncio.gather(task, return_exceptions=True)

    assert task.cancelled()
    assert coordinator._active_operations == set()
    assert coordinator._background_tasks == set()
    assert coordinator.progress.status is shutdown.ShutdownStatus.FORCED


@pytest.mark.unit
@pytest.mark.asyncio
async def test_finalization_records_internal_cleanup_errors() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config())

    class BrokenSet(set[str]):
        def clear(self) -> None:
            raise RuntimeError("operation registry is unavailable")

    coordinator._active_operations = BrokenSet()

    await coordinator._finalize_shutdown()

    assert coordinator.progress.errors == [
        "Finalization error: operation registry is unavailable"
    ]


@pytest.mark.unit
def test_signal_handler_setup_registers_handlers_and_runs_shutdown_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: dict[int, Callable[[int, Any], None]] = {}
    executions: list[Any] = []
    timestamps = iter([100.0, 100.0, 102.0])

    class Loop:
        def is_running(self) -> bool:
            return False

        def run_until_complete(self, awaitable: Any) -> None:
            executions.append("run")
            awaitable.close()

    loop = Loop()

    def signal_handler(signum: int, handler: Callable[[int, Any], None]) -> None:
        registered[signum] = handler

    async def initiate_shutdown() -> bool:
        executions.append("shutdown")
        return True

    def wait_for(awaitable: Any, *, timeout: float) -> _CloseableAwaitable:
        assert timeout == 5.0
        awaitable.close()
        return _CloseableAwaitable()

    signal_module = SimpleNamespace(
        SIGTERM=15,
        SIGINT=2,
        SIGHUP=1,
        signal=signal_handler,
    )
    monkeypatch.setattr(shutdown, "signal", signal_module)
    monkeypatch.setattr(
        shutdown, "time", SimpleNamespace(time=lambda: next(timestamps))
    )
    monkeypatch.setattr(
        shutdown,
        "asyncio",
        SimpleNamespace(get_event_loop=lambda: loop, wait_for=wait_for),
    )
    monkeypatch.setattr(
        shutdown,
        "shutdown_coordinator",
        SimpleNamespace(initiate_shutdown=initiate_shutdown),
    )

    shutdown.setup_signal_handlers()

    assert set(registered) == {
        signal_module.SIGTERM,
        signal_module.SIGINT,
        signal_module.SIGHUP,
    }
    registered[signal_module.SIGTERM](signal_module.SIGTERM, None)
    registered[signal_module.SIGINT](signal_module.SIGINT, None)
    registered[signal_module.SIGINT](signal_module.SIGINT, None)
    assert executions == ["run", "run", "run"]


@pytest.mark.unit
def test_signal_handler_creates_loop_when_no_current_loop_and_schedules_when_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: dict[int, Callable[[int, Any], None]] = {}
    created_loops: list[Any] = []
    scheduled: list[str] = []

    class Loop:
        def __init__(self, running: bool) -> None:
            self.running = running
            self.executions = 0

        def is_running(self) -> bool:
            return self.running

        def run_until_complete(self, awaitable: Any) -> None:
            self.executions += 1
            awaitable.close()

    non_running = Loop(False)
    running = Loop(True)

    def get_event_loop() -> Any:
        raise RuntimeError("no active loop")

    def new_event_loop() -> Loop:
        created_loops.append(non_running)
        return non_running

    def create_task(coroutine: Any) -> None:
        scheduled.append("shutdown task")
        coroutine.close()

    async def initiate_shutdown() -> bool:
        return True

    def wait_for(awaitable: Any, **_kwargs: Any) -> _CloseableAwaitable:
        awaitable.close()
        return _CloseableAwaitable()

    signal_module = SimpleNamespace(
        SIGTERM=15, SIGINT=2, signal=lambda n, h: registered.update({n: h})
    )
    monkeypatch.setattr(shutdown, "signal", signal_module)
    monkeypatch.setattr(
        shutdown,
        "asyncio",
        SimpleNamespace(
            get_event_loop=get_event_loop,
            new_event_loop=new_event_loop,
            set_event_loop=lambda loop: created_loops.append(loop),
            wait_for=wait_for,
            create_task=create_task,
        ),
    )
    monkeypatch.setattr(
        shutdown,
        "shutdown_coordinator",
        SimpleNamespace(initiate_shutdown=initiate_shutdown),
    )

    shutdown.setup_signal_handlers()
    registered[signal_module.SIGTERM](signal_module.SIGTERM, None)
    assert non_running.executions == 1
    assert created_loops == [non_running, non_running]

    # Use a running loop on the second setup call; the scheduled coroutine is closed
    # by the fake create_task, so the test never starts or exits the process.
    monkeypatch.setattr(
        shutdown,
        "asyncio",
        SimpleNamespace(
            get_event_loop=lambda: running,
            create_task=create_task,
        ),
    )
    shutdown.setup_signal_handlers()
    registered[signal_module.SIGTERM](signal_module.SIGTERM, None)
    assert scheduled == ["shutdown task"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_status_failure_path_sets_completion_time() -> None:
    coordinator = shutdown.ShutdownCoordinator(_config())

    async def fail_sequence(save_state: bool) -> bool:
        del save_state
        return False

    coordinator._execute_shutdown_sequence = fail_sequence  # type: ignore[method-assign]

    assert await coordinator.initiate_shutdown(save_state=False) is False
    status = coordinator.get_shutdown_status()
    assert status["status"] == "failed"
    assert status["phase"] == "failed"
    assert status["started_at"] is not None
    assert status["completed_at"] is not None
