from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one marker, found {count}")
    return text.replace(old, new, 1)


worker_path = Path("packages/neuros-core/src/neuros/runtime/process_worker.py")
text = worker_path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "class ProcessWorkerSerializationError(ProcessWorkerError):\n    pass\n\n\nclass ProcessWorkerRemoteError(ProcessWorkerError):\n",
    "class ProcessWorkerSerializationError(ProcessWorkerError):\n    pass\n\n\nclass ProcessWorkerTerminationError(ProcessWorkerError):\n    pass\n\n\nclass ProcessWorkerRemoteError(ProcessWorkerError):\n",
    "termination error type",
)
old_shutdown = '''    def abort(self) -> None:
        self._terminal = True
        conn, process = self._conn, self._process
        self._conn = self._process = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if process is None:
            return
        if process.is_alive():
            process.terminate()
            process.join(self.termination_grace_s)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(self.termination_grace_s)
        if not process.is_alive():
            process.close()

    def close(self) -> None:
        self._terminal = True
        conn, process = self._conn, self._process
        if process is None:
            if conn is not None:
                conn.close()
            self._conn = None
            return
        if process.is_alive() and conn is not None:
            try:
                self._send_control("shutdown")
                if conn.poll(self.termination_grace_s):
                    response = pickle.loads(conn.recv_bytes())
                    self._identity(response, None)
            except Exception:
                pass
        process.join(self.termination_grace_s)
        if process.is_alive():
            process.terminate()
            process.join(self.termination_grace_s)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(self.termination_grace_s)
        if conn is not None:
            conn.close()
        self._conn = self._process = None
        if not process.is_alive():
            process.close()
'''
new_shutdown = '''    def _terminate_owned_process(self, process: mp.Process) -> None:
        """Prove the direct child is dead or fail closed with its handle retained."""

        if process.is_alive():
            process.terminate()
            process.join(self.termination_grace_s)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(self.termination_grace_s)
        if process.is_alive():
            self._process = process
            raise ProcessWorkerTerminationError(
                self.node_id,
                "direct child remained alive after terminate/join/kill escalation",
            )
        process.close()
        self._process = None

    def abort(self) -> None:
        self._terminal = True
        conn, process = self._conn, self._process
        self._conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if process is None:
            self._process = None
            return
        self._terminate_owned_process(process)

    def close(self) -> None:
        self._terminal = True
        conn, process = self._conn, self._process
        if process is None:
            if conn is not None:
                conn.close()
            self._conn = None
            return
        if process.is_alive() and conn is not None:
            try:
                self._send_control("shutdown")
                if conn.poll(self.termination_grace_s):
                    response = pickle.loads(conn.recv_bytes())
                    self._identity(response, None)
            except Exception:
                pass
        process.join(self.termination_grace_s)
        if conn is not None:
            conn.close()
        self._conn = None
        if process.is_alive():
            self._terminate_owned_process(process)
            return
        process.close()
        self._process = None
'''
text = replace_once(text, old_shutdown, new_shutdown, "worker terminal authority")
worker_path.write_text(text, encoding="utf-8")


executor_path = Path("packages/neuros-core/src/neuros/runtime/executor.py")
text = executor_path.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''class RuntimeUnexpectedCancellationError(RuntimeError):
''',
    '''class RuntimeProcessCleanupError(RuntimeError):
    """Raised when executor-owned process cleanup cannot prove termination."""

    def __init__(self, failures: tuple[tuple[str, str, str], ...]) -> None:
        self.failures = tuple(sorted(failures))
        detail = "; ".join(
            f"{node_id}: {error_type}: {message}"
            for node_id, error_type, message in self.failures
        )
        super().__init__(f"process cleanup failed: {detail}")


class RuntimeUnexpectedCancellationError(RuntimeError):
''',
    "runtime cleanup error type",
)
text = replace_once(
    text,
    "        self._stopping = False\n        self._build_channels()\n",
    "        self._stopping = False\n        self._process_cleanup_error: RuntimeProcessCleanupError | None = None\n        self._build_channels()\n",
    "cleanup error state",
)
old_close = '''    async def _close_process_workers(self) -> None:
        """Close every executor-owned direct child before supervision completes."""

        workers = tuple(self._process_workers.values())
        if not workers:
            return
        await asyncio.gather(
            *(asyncio.to_thread(worker.close) for worker in workers),
            return_exceptions=True,
        )
'''
new_close = '''    async def _close_process_workers(self) -> None:
        """Prove every executor-owned child is closed before successful terminal state."""

        workers = tuple(sorted(self._process_workers.items()))
        if not workers:
            return
        results = await asyncio.gather(
            *(asyncio.to_thread(worker.close) for _, worker in workers),
            return_exceptions=True,
        )
        failures = tuple(
            (
                node_id,
                type(result).__name__,
                str(result),
            )
            for (node_id, _), result in zip(workers, results)
            if isinstance(result, BaseException)
        )
        if not failures:
            return
        if self._process_cleanup_error is not None:
            return
        error = RuntimeProcessCleanupError(failures)
        self._process_cleanup_error = error
        metadata = {
            "error_type": type(error).__name__,
            "message": str(error),
            "failures": error.failures,
        }
        if self.failure is None:
            self.failure = RuntimeFailure(
                "runtime", type(error).__name__, str(error)
            )
            self.stopped_ns = time.monotonic_ns()
            await self._transition(
                RuntimeState.FAILED,
                "runtime_process_cleanup_failed",
                **metadata,
            )
            return
        await self._event_queue.put(
            RuntimeEvent(
                event="runtime_process_cleanup_failed",
                state=RuntimeState.FAILED,
                metadata=metadata,
            )
        )
'''
text = replace_once(text, old_close, new_close, "strict process cleanup")
text = replace_once(
    text,
    '''    async def _finish_successfully(self) -> None:
        if self.state not in (RuntimeState.DRAINING, RuntimeState.STOPPED):
            await self._transition(RuntimeState.DRAINING, "runtime_draining")
        self.stopped_ns = time.monotonic_ns()
        if self.state is not RuntimeState.STOPPED:
            await self._transition(RuntimeState.STOPPED, "runtime_stopped")
''',
    '''    async def _finish_successfully(self) -> None:
        if self.state not in (RuntimeState.DRAINING, RuntimeState.STOPPED):
            await self._transition(RuntimeState.DRAINING, "runtime_draining")
        await self._close_process_workers()
        if self.failure is not None:
            return
        self.stopped_ns = time.monotonic_ns()
        if self.state is not RuntimeState.STOPPED:
            await self._transition(RuntimeState.STOPPED, "runtime_stopped")
''',
    "cleanup before stopped",
)
text = replace_once(
    text,
    '''        if self.state is RuntimeState.CREATED:
            self.stopped_ns = time.monotonic_ns()
            await self._transition(RuntimeState.STOPPED, "runtime_stopped")
            await self._close_process_workers()
            return
''',
    '''        if self.state is RuntimeState.CREATED:
            await self._close_process_workers()
            if self.failure is None:
                self.stopped_ns = time.monotonic_ns()
                await self._transition(RuntimeState.STOPPED, "runtime_stopped")
            return
''',
    "created cleanup ordering",
)
executor_path.write_text(text, encoding="utf-8")


init_path = Path("packages/neuros-core/src/neuros/runtime/__init__.py")
text = init_path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "    RuntimeFailure,\n    RuntimeUnexpectedCancellationError,\n",
    "    RuntimeFailure,\n    RuntimeProcessCleanupError,\n    RuntimeUnexpectedCancellationError,\n",
    "runtime cleanup export import",
)
text = replace_once(
    text,
    "    ProcessWorkerSerializationError,\n    ProcessWorkerTimeoutError,\n",
    "    ProcessWorkerSerializationError,\n    ProcessWorkerTerminationError,\n    ProcessWorkerTimeoutError,\n",
    "termination export import",
)
text = replace_once(
    text,
    '    "ProcessWorkerSerializationError",\n    "ProcessWorkerTimeoutError",\n',
    '    "ProcessWorkerSerializationError",\n    "ProcessWorkerTerminationError",\n    "ProcessWorkerTimeoutError",\n',
    "termination all",
)
text = replace_once(
    text,
    '    "RuntimeFailure",\n    "RuntimeGraph",\n',
    '    "RuntimeFailure",\n    "RuntimeProcessCleanupError",\n    "RuntimeGraph",\n',
    "cleanup all",
)
init_path.write_text(text, encoding="utf-8")


process_test_path = Path("tests/test_runtime_process_authority.py")
text = process_test_path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "from neuros.runtime.process_worker import PersistentProcessWorker, ProcessWorkerProtocolError\n",
    "from neuros.runtime.process_worker import (\n    PersistentProcessWorker,\n    ProcessWorkerProtocolError,\n    ProcessWorkerTerminationError,\n)\n",
    "termination test import",
)
insert = '''\n\nclass ImmortalProcess:\n    def __init__(self):\n        self.terminate_calls = 0\n        self.kill_calls = 0\n        self.join_calls = 0\n\n    def is_alive(self):\n        return True\n\n    def terminate(self):\n        self.terminate_calls += 1\n\n    def kill(self):\n        self.kill_calls += 1\n\n    def join(self, timeout=None):\n        self.join_calls += 1\n\n    def close(self):\n        raise AssertionError("live process handle must not be closed")\n\n\ndef test_direct_child_surviving_escalation_fails_closed():\n    worker = PersistentProcessWorker(\n        "transform", CounterTransform(), execution_timeout_s=1.0\n    )\n    process = ImmortalProcess()\n    worker._process = process\n\n    with pytest.raises(ProcessWorkerTerminationError, match="remained alive"):\n        worker.abort()\n\n    assert process.terminate_calls == 1\n    assert process.kill_calls == 1\n    assert process.join_calls == 2\n    assert worker._process is process\n'''
if "def test_direct_child_surviving_escalation_fails_closed" in text:
    raise SystemExit("termination adversary already present")
text += insert
process_test_path.write_text(text, encoding="utf-8")


fault_test_path = Path("tests/test_runtime_fault_qualification.py")
text = fault_test_path.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''class FakeProcessWorker:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
''',
    '''class FakeProcessWorker:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class FailingProcessWorker:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        raise RuntimeError("worker cleanup refused")
''',
    "failing cleanup worker",
)
insert = '''\n\n@pytest.mark.asyncio\nasync def test_process_cleanup_failure_prevents_false_stopped_state():\n    executor = RuntimeExecutor(\n        source_sink_graph(FiniteSource([1]), CollectingSink())\n    )\n    failing_worker = FailingProcessWorker()\n    executor._process_workers["stuck"] = failing_worker\n\n    with pytest.raises(RuntimeError, match="RuntimeProcessCleanupError"):\n        await executor.run()\n\n    snapshot = executor.snapshot()\n    assert snapshot["state"] == "failed"\n    assert snapshot["failure"]["node_id"] == "runtime"\n    assert snapshot["failure"]["error_type"] == "RuntimeProcessCleanupError"\n    assert "worker cleanup refused" in snapshot["failure"]["message"]\n    events = await collect_events(executor)\n    assert sum(event.event == "runtime_process_cleanup_failed" for event in events) == 1\n    assert not any(event.event == "runtime_stopped" for event in events)\n\n\n@pytest.mark.asyncio\nasync def test_cleanup_failure_does_not_overwrite_primary_node_culprit():\n    executor = RuntimeExecutor(\n        source_transform_sink_graph(\n            FiniteSource([1]),\n            FailTransform(),\n            CollectingSink(),\n        )\n    )\n    failing_worker = FailingProcessWorker()\n    executor._process_workers["stuck"] = failing_worker\n\n    with pytest.raises(RuntimeError, match="qualified transform failure"):\n        await executor.run()\n\n    assert executor.snapshot()["failure"] == {\n        "node_id": "transform",\n        "error_type": "ValueError",\n        "message": "qualified transform failure",\n    }\n    events = await collect_events(executor)\n    assert sum(event.event == "runtime_process_cleanup_failed" for event in events) == 1\n'''
if "test_process_cleanup_failure_prevents_false_stopped_state" in text:
    raise SystemExit("cleanup adversaries already present")
text += insert
fault_test_path.write_text(text, encoding="utf-8")
