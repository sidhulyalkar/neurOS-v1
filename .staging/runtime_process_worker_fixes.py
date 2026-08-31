from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one marker, found {count}")
    return text.replace(old, new, 1)


worker_path = Path("packages/neuros-core/src/neuros/runtime/process_worker.py")
worker = worker_path.read_text(encoding="utf-8")
worker = replace_once(
    worker,
    '''            try:
                request = pickle.loads(conn.recv_bytes())
            except EOFError:
                return
            except Exception as exc:
                _send(conn, {**base, "kind": "protocol_error", "message": str(exc)})
                return
''',
    '''            try:
                payload = conn.recv_bytes()
            except (EOFError, ConnectionResetError, BrokenPipeError, OSError):
                return
            try:
                request = pickle.loads(payload)
            except Exception as exc:
                try:
                    _send(
                        conn,
                        {**base, "kind": "protocol_error", "message": str(exc)},
                    )
                except (EOFError, ConnectionResetError, BrokenPipeError, OSError):
                    pass
                return
''',
    "expected parent IPC teardown",
)
worker = replace_once(
    worker,
    '''    def _send_control(self, command: str) -> None:
        if self._conn is None:
            raise ProcessWorkerCrashedError(self.node_id, "worker IPC is closed")
        _send(
            self._conn,
            {
                "protocol": _PROTOCOL,
                "node_id": self.node_id,
                "generation": self.generation,
                "command": command,
            },
        )
''',
    '''    def _send_control(self, command: str) -> None:
        if self._conn is None:
            raise ProcessWorkerCrashedError(self.node_id, "worker IPC is closed")
        try:
            _send(
                self._conn,
                {
                    "protocol": _PROTOCOL,
                    "node_id": self.node_id,
                    "generation": self.generation,
                    "command": command,
                },
            )
        except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as exc:
            raise ProcessWorkerCrashedError(
                self.node_id, f"worker IPC failed while sending {command}"
            ) from exc
''',
    "control-send crash classification",
)
worker = replace_once(
    worker,
    '''                _send(
                    self._conn,
                    {
                        "protocol": _PROTOCOL,
                        "node_id": self.node_id,
                        "generation": self.generation,
                        "command": "call",
                        "request_id": request_id,
                        "method": method,
                        "item": item_bytes,
                    },
                )
                response = await asyncio.to_thread(
                    self._recv, self.execution_timeout_s, request_id
                )
''',
    '''                try:
                    await asyncio.to_thread(
                        _send,
                        self._conn,
                        {
                            "protocol": _PROTOCOL,
                            "node_id": self.node_id,
                            "generation": self.generation,
                            "command": "call",
                            "request_id": request_id,
                            "method": method,
                            "item": item_bytes,
                        },
                    )
                except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as exc:
                    raise ProcessWorkerCrashedError(
                        self.node_id,
                        f"worker IPC failed while sending request {request_id}",
                    ) from exc
                response = await asyncio.to_thread(
                    self._recv, self.execution_timeout_s, request_id
                )
                if not isinstance(response, dict):
                    raise ProcessWorkerProtocolError(
                        self.node_id, "worker response is not a mapping"
                    )
                self._identity(response, request_id)
''',
    "request send and second identity gate",
)
worker = replace_once(
    worker,
    '''    async def heartbeat(self) -> None:
        async with self._lock:
            try:
                await asyncio.to_thread(self._start)
                await asyncio.to_thread(self._send_control, "heartbeat")
                response = await asyncio.to_thread(self._recv, self.startup_timeout_s, None)
                if response.get("kind") != "heartbeat":
                    raise ProcessWorkerProtocolError(self.node_id, "worker heartbeat failed")
            except Exception:
                await asyncio.to_thread(self.abort)
                raise
''',
    '''    async def heartbeat(self) -> None:
        async with self._lock:
            try:
                await asyncio.to_thread(self._start)
                await asyncio.to_thread(self._send_control, "heartbeat")
                response = await asyncio.to_thread(
                    self._recv, self.startup_timeout_s, None
                )
                if response.get("kind") != "heartbeat":
                    raise ProcessWorkerProtocolError(self.node_id, "worker heartbeat failed")
            except asyncio.CancelledError:
                await asyncio.shield(asyncio.to_thread(self.abort))
                raise
            except Exception:
                await asyncio.to_thread(self.abort)
                raise
''',
    "heartbeat cancellation authority",
)
worker_path.write_text(worker, encoding="utf-8")


fault_path = Path("tests/test_runtime_fault_qualification.py")
fault = fault_path.read_text(encoding="utf-8")
fault = replace_once(
    fault,
    '''class FakeProcessPool:
    def __init__(self):
        self.shutdown_calls = []

    def shutdown(self, *, wait, cancel_futures):
        self.shutdown_calls.append(
            {"wait": wait, "cancel_futures": cancel_futures}
        )
''',
    '''class FakeProcessWorker:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
''',
    "legacy pool test double",
)
fault = replace_once(
    fault,
    '''@pytest.mark.asyncio
async def test_executor_owned_process_pool_is_closed_on_failure_path():
    executor = RuntimeExecutor(
        source_transform_sink_graph(
            FiniteSource([1]),
            FailTransform(),
            CollectingSink(),
        )
    )
    fake_pool = FakeProcessPool()
    executor._process_pool = fake_pool  # ownership-path test, no subprocess spawned

    with pytest.raises(RuntimeError, match="qualified transform failure"):
        await executor.run()

    assert fake_pool.shutdown_calls == [
        {"wait": False, "cancel_futures": True}
    ]
    assert executor._process_pool is None
    assert_executor_tasks_terminal(executor)
''',
    '''@pytest.mark.asyncio
async def test_executor_owned_process_workers_are_closed_on_failure_path():
    executor = RuntimeExecutor(
        source_transform_sink_graph(
            FiniteSource([1]),
            FailTransform(),
            CollectingSink(),
        )
    )
    fake_worker = FakeProcessWorker()
    executor._process_workers["owned-test-worker"] = fake_worker

    with pytest.raises(RuntimeError, match="qualified transform failure"):
        await executor.run()

    assert fake_worker.close_calls == 1
    assert_executor_tasks_terminal(executor)
''',
    "legacy pooled-process ownership test",
)
fault_path.write_text(fault, encoding="utf-8")
