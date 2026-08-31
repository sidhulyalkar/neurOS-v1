"""Native execution engine for :class:`neuros.runtime.RuntimeGraph`.

The executor is intentionally small and explicit. It owns queue/backpressure
semantics, lifecycle, failure propagation, node scheduling, latency telemetry,
and the live/replay symmetry of the neurOS data plane. Concrete hardware,
models, storage backends, and ORION implementations remain outside the kernel.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, AsyncIterator, Callable

import numpy as np

from neuros.contracts import DecoderOutput, NeuralWindow, SignalFrame, TransformEmission
from neuros.errors import ProcessingError
from neuros.runtime.graph import NodeKind, RuntimeEdge, RuntimeGraph, RuntimeNode
from neuros.runtime.process_worker import PersistentProcessWorker
from neuros.runtime.shared_process_worker import SharedMemoryProcessWorker
from neuros.runtime.lifecycle import RuntimeEvent, RuntimeState
from neuros.runtime.queues import OverflowPolicy, QueueStats, put_with_policy


class ExecutionClass(str, Enum):
    """Execution isolation requested by a runtime node.

    ``GPU`` is scheduled inline because the operator owns its framework/device
    context; the label exists so graph specifications and telemetry preserve the
    scheduling intent. ``PROCESS`` owns one persistent direct child per runtime
    node and requires an explicit hard execution timeout.
    """

    INLINE = "inline"
    THREAD = "thread"
    PROCESS = "process"
    GPU = "gpu"


@dataclass(slots=True)
class NodeStats:
    processed: int = 0
    failed: int = 0
    total_latency_ns: int = 0
    max_latency_ns: int = 0
    _recent_latency_ns: deque[int] = field(
        default_factory=lambda: deque(maxlen=4096), repr=False
    )

    def observe(self, latency_ns: int) -> None:
        self.processed += 1
        self.total_latency_ns += latency_ns
        self.max_latency_ns = max(self.max_latency_ns, latency_ns)
        self._recent_latency_ns.append(latency_ns)

    def snapshot(self) -> dict[str, float | int]:
        values = sorted(self._recent_latency_ns)

        def percentile(q: float) -> float:
            if not values:
                return 0.0
            index = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
            return values[index] / 1_000_000.0

        mean_ms = (
            self.total_latency_ns / self.processed / 1_000_000.0
            if self.processed
            else 0.0
        )
        return {
            "processed": self.processed,
            "failed": self.failed,
            "mean_latency_ms": mean_ms,
            "p50_latency_ms": percentile(0.50),
            "p95_latency_ms": percentile(0.95),
            "p99_latency_ms": percentile(0.99),
            "max_latency_ms": self.max_latency_ns / 1_000_000.0,
        }


@dataclass(slots=True)
class EdgeChannel:
    edge: RuntimeEdge
    queue: asyncio.Queue[Any]
    policy: OverflowPolicy
    stats: QueueStats = field(default_factory=QueueStats)


@dataclass(frozen=True, slots=True)
class RuntimeFailure:
    node_id: str
    error_type: str
    message: str


class RuntimeDrainTimeoutError(RuntimeError):
    """Raised when a requested runtime drain cannot terminate in time.

    A drain timeout is a runtime failure, not a successful stop. The pending
    node identifiers are captured in deterministic lexical order so callers can
    persist a stable failure record without depending on task object reprs.
    """

    def __init__(self, timeout_s: float, pending_node_ids: tuple[str, ...]) -> None:
        self.timeout_s = float(timeout_s)
        self.pending_node_ids = tuple(sorted(pending_node_ids))
        pending = ", ".join(self.pending_node_ids) if self.pending_node_ids else "<none>"
        super().__init__(
            f"runtime drain exceeded {self.timeout_s:.6f}s; pending nodes: {pending}"
        )


class RuntimeProcessCleanupError(RuntimeError):
    """Raised when executor-owned process cleanup cannot prove termination."""

    def __init__(self, failures: tuple[tuple[str, str, str], ...]) -> None:
        self.failures = tuple(sorted(failures))
        detail = "; ".join(
            f"{node_id}: {error_type}: {message}"
            for node_id, error_type, message in self.failures
        )
        super().__init__(f"process cleanup failed: {detail}")


class RuntimeUnexpectedCancellationError(RuntimeError):
    """Raised when a runtime node task is cancelled without runtime authority."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        super().__init__(
            f"runtime node task {node_id!r} was cancelled without shutdown authority"
        )


class _AttributedNodeError(RuntimeError):
    """Internal wrapper for failures raised while executing another node's task."""

    def __init__(self, node_id: str, cause: Exception) -> None:
        self.node_id = node_id
        self.cause = cause
        super().__init__(str(cause))


class _Stop:
    __slots__ = ()


_STOP = _Stop()


def _call(func: Callable[[Any], Any], item: Any) -> Any:
    """Execute a sync or async callback inside the current execution domain."""
    result = func(item)
    if not inspect.isawaitable(result):
        return result

    async def await_result() -> Any:
        return await result

    return asyncio.run(await_result())


_ProcessWorker = PersistentProcessWorker | SharedMemoryProcessWorker


class RuntimeExecutor:
    """Execute a validated :class:`RuntimeGraph`.

    The runtime supports finite replay sources and indefinitely streaming live
    sources with the same graph. Each edge owns a bounded queue and explicit
    overflow policy. A source completion sentinel is propagated through the
    graph so finite experiments terminate deterministically without cancellation.
    """

    def __init__(
        self,
        graph: RuntimeGraph,
        *,
        drain_timeout_s: float = 2.0,
    ) -> None:
        if drain_timeout_s <= 0:
            raise ValueError("drain_timeout_s must be positive")
        graph.validate()
        self.graph = graph
        self.drain_timeout_s = drain_timeout_s
        self.state = RuntimeState.CREATED
        self.failure: RuntimeFailure | None = None
        self.started_ns: int | None = None
        self.stopped_ns: int | None = None
        self._channels: dict[tuple[str, str], EdgeChannel] = {}
        self._incoming: dict[str, list[EdgeChannel]] = {
            node_id: [] for node_id in graph.nodes
        }
        self._outgoing: dict[str, list[EdgeChannel]] = {
            node_id: [] for node_id in graph.nodes
        }
        self._node_stats = {node_id: NodeStats() for node_id in graph.nodes}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._completion_task: asyncio.Task[None] | None = None
        self._output_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._event_queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue()
        self._process_workers: dict[str, _ProcessWorker] = {}
        self._process_receipts: dict[str, deque[dict[str, Any]]] = {
            node_id: deque(maxlen=4096)
            for node_id, node in graph.nodes.items()
            if node.executor == "process"
        }
        self._stopping = False
        self._process_cleanup_error: RuntimeProcessCleanupError | None = None
        self._build_channels()

    def _build_channels(self) -> None:
        for edge in self.graph.edges:
            key = (edge.source, edge.target)
            if key in self._channels:
                raise ValueError(f"Duplicate runtime edge: {edge.source} -> {edge.target}")
            channel = EdgeChannel(
                edge=edge,
                queue=asyncio.Queue(maxsize=edge.capacity),
                policy=OverflowPolicy(edge.overflow),
            )
            self._channels[key] = channel
            self._outgoing[edge.source].append(channel)
            self._incoming[edge.target].append(channel)

    async def _transition(self, state: RuntimeState, event: str, **metadata: Any) -> None:
        self.state = state
        await self._event_queue.put(
            RuntimeEvent(event=event, state=state, metadata=metadata)
        )

    @property
    def node_stats(self) -> dict[str, NodeStats]:
        return self._node_stats

    @property
    def edge_channels(self) -> dict[tuple[str, str], EdgeChannel]:
        return self._channels

    async def _close_process_workers(self) -> None:
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

    async def start(self) -> None:
        if self.state is not RuntimeState.CREATED:
            raise RuntimeError(
                f"Runtime can only start from CREATED, got {self.state.value}"
            )
        await self._transition(RuntimeState.STARTING, "runtime_starting")
        self.started_ns = time.monotonic_ns()

        for node in self.graph.nodes.values():
            if node.kind is NodeKind.MONITOR:
                continue
            self._tasks[node.node_id] = asyncio.create_task(
                self._run_guarded(node), name=f"neuros:{node.node_id}"
            )

        self._completion_task = asyncio.create_task(
            self._supervise(), name="neuros:supervisor"
        )
        await self._transition(RuntimeState.RUNNING, "runtime_running")

    async def _supervise(self) -> None:
        try:
            if not self._tasks:
                await self._finish_successfully()
                return
            results = await asyncio.gather(
                *self._tasks.values(), return_exceptions=True
            )
            if self.failure is not None:
                self.stopped_ns = time.monotonic_ns()
                if self.state is not RuntimeState.FAILED:
                    await self._transition(RuntimeState.FAILED, "runtime_failed")
                return
            unexpected = [
                result
                for result in results
                if isinstance(result, BaseException)
                and not isinstance(result, asyncio.CancelledError)
            ]
            if unexpected and not self._stopping:
                exc = unexpected[0]
                self.failure = RuntimeFailure(
                    "unknown", type(exc).__name__, str(exc)
                )
                self.stopped_ns = time.monotonic_ns()
                await self._transition(RuntimeState.FAILED, "runtime_failed")
                return
            await self._finish_successfully()
        finally:
            await self._close_process_workers()

    async def _finish_successfully(self) -> None:
        if self.state not in (RuntimeState.DRAINING, RuntimeState.STOPPED):
            await self._transition(RuntimeState.DRAINING, "runtime_draining")
        await self._close_process_workers()
        if self.failure is not None:
            return
        self.stopped_ns = time.monotonic_ns()
        if self.state is not RuntimeState.STOPPED:
            await self._transition(RuntimeState.STOPPED, "runtime_stopped")

    async def _run_guarded(self, node: RuntimeNode) -> None:
        try:
            if node.kind is NodeKind.SOURCE:
                await self._run_source(node)
            elif node.kind is NodeKind.FUSION:
                await self._run_fusion(node)
            else:
                await self._run_unary(node)
        except asyncio.CancelledError:
            if self.failure is None and not self._stopping:
                cancellation = RuntimeUnexpectedCancellationError(node.node_id)
                self._node_stats[node.node_id].failed += 1
                self.failure = RuntimeFailure(
                    node.node_id,
                    type(cancellation).__name__,
                    str(cancellation),
                )
                await self._transition(
                    RuntimeState.FAILED,
                    "node_cancelled",
                    node_id=node.node_id,
                    error_type=type(cancellation).__name__,
                    message=str(cancellation),
                )
                for peer_id, task in self._tasks.items():
                    if peer_id != node.node_id and not task.done():
                        task.cancel()
            raise
        except Exception as exc:
            culprit_id = node.node_id
            cause = exc
            if isinstance(exc, _AttributedNodeError):
                culprit_id = exc.node_id
                cause = exc.cause
            self._node_stats[culprit_id].failed += 1
            error_type = str(
                getattr(cause, "runtime_error_type", type(cause).__name__)
            )
            if self.failure is None:
                self.failure = RuntimeFailure(culprit_id, error_type, str(cause))
                await self._transition(
                    RuntimeState.FAILED,
                    "node_failed",
                    node_id=culprit_id,
                    error_type=error_type,
                    message=str(cause),
                )
                # The task currently executing this handler must be allowed to
                # finish propagating its exception. All peers are cancelled.
                for peer_id, task in self._tasks.items():
                    if peer_id != node.node_id and not task.done():
                        task.cancel()
            raise

    async def _run_source(self, node: RuntimeNode) -> None:
        source = node.operator
        if not all(hasattr(source, name) for name in ("start", "stop", "frames")):
            raise TypeError(f"Source node {node.node_id} does not implement Source")
        await source.start()
        completed_normally = False
        cancelled = False
        try:
            async for item in source.frames():
                await self._emit(node, item)
            completed_normally = True
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            await source.stop()
            # Only normal completion or an explicit graceful stop owns an
            # in-band shutdown marker. During failure propagation the peer
            # consumer may already be cancelled, so enqueueing _STOP into a
            # saturated data queue can deadlock containment indefinitely.
            graceful_stop = (
                cancelled and self._stopping and self.failure is None
            )
            if completed_normally or graceful_stop:
                await self._emit_stop(node.node_id)

    async def _run_unary(self, node: RuntimeNode) -> None:
        incoming = self._incoming[node.node_id]
        if len(incoming) != 1:
            raise ValueError(
                f"{node.kind.value} node {node.node_id} requires exactly one input; "
                f"got {len(incoming)}"
            )
        channel = incoming[0]
        while True:
            item = await channel.queue.get()
            channel.queue.task_done()
            if item is _STOP:
                await self._emit_stop(node.node_id)
                return
            started = time.perf_counter_ns()
            result = await self._invoke(node, item)
            self._node_stats[node.node_id].observe(
                time.perf_counter_ns() - started
            )
            if result is None or node.kind is NodeKind.SINK:
                continue
            if isinstance(result, TransformEmission):
                for emitted in result.items:
                    await self._emit(node, emitted)
            else:
                await self._emit(node, result)

    async def _run_fusion(self, node: RuntimeNode) -> None:
        incoming = self._incoming[node.node_id]
        if len(incoming) < 2:
            raise ValueError(
                f"Fusion node {node.node_id} requires at least two inputs"
            )
        latest: dict[str, Any] = {}
        closed: set[str] = set()
        sequence_id = 0
        while len(closed) < len(incoming):
            pending = {
                asyncio.create_task(
                    channel.queue.get(),
                    name=f"neuros:{node.node_id}:input:{channel.edge.source}",
                ): channel
                for channel in incoming
                if channel.edge.source not in closed
            }
            if not pending:
                break
            try:
                done, not_done = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
            except BaseException:
                # Fusion owns these temporary queue-get tasks. If the fusion
                # node is cancelled, every child must be cancelled and awaited
                # before the node itself can become terminal.
                for task in pending:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise
            for task in not_done:
                task.cancel()
            for task in not_done:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            for task in done:
                channel = pending[task]
                item = task.result()
                channel.queue.task_done()
                source_id = channel.edge.source
                if item is _STOP:
                    closed.add(source_id)
                    continue
                latest[source_id] = item
                if len(latest) == len(incoming):
                    started = time.perf_counter_ns()
                    result = await self._fuse(node, latest, sequence_id)
                    self._node_stats[node.node_id].observe(
                        time.perf_counter_ns() - started
                    )
                    sequence_id += 1
                    await self._emit(node, result)
        await self._emit_stop(node.node_id)

    async def _fuse(
        self, node: RuntimeNode, latest: dict[str, Any], sequence_id: int
    ) -> Any:
        if node.operator is not None and hasattr(node.operator, "fuse"):
            return await self._invoke_callable(
                node, node.operator.fuse, dict(latest)
            )

        values = list(latest.values())
        if all(isinstance(item, SignalFrame) for item in values):
            frames = [item for item in values if isinstance(item, SignalFrame)]
            data = np.concatenate(
                [np.asarray(frame.data).reshape(-1) for frame in frames]
            )
            reference = max(frames, key=lambda frame: frame.timestamp_ns)
            return replace(
                reference,
                stream_id=str(node.metadata.get("stream_id", node.node_id)),
                sequence_id=sequence_id,
                data=data,
                synchronized_time_ns=max(
                    (frame.synchronized_time_ns or frame.timestamp_ns)
                    for frame in frames
                ),
                metadata={
                    **dict(reference.metadata),
                    "fused_sources": tuple(frame.stream_id for frame in frames),
                },
            )
        return np.concatenate([np.asarray(item).reshape(-1) for item in values])

    async def _invoke(self, node: RuntimeNode, item: Any) -> Any:
        operator = node.operator
        if node.kind is NodeKind.TRANSFORM:
            if not hasattr(operator, "transform"):
                raise TypeError(
                    f"Transform node {node.node_id} lacks transform()"
                )
            return await self._invoke_callable(node, operator.transform, item)
        if node.kind is NodeKind.DECODER:
            if not hasattr(operator, "infer"):
                raise TypeError(f"Decoder node {node.node_id} lacks infer()")
            if isinstance(item, NeuralWindow):
                value = item.as_batch()
            else:
                value = np.asarray(
                    item.data if isinstance(item, SignalFrame) else item
                )
                if value.ndim == 1:
                    value = value.reshape(1, -1)
            result = await self._invoke_callable(node, operator.infer, value)
            if isinstance(item, NeuralWindow) and isinstance(result, DecoderOutput):
                result = replace(
                    result,
                    metadata={
                        **dict(result.metadata),
                        "neuros_stream_id": item.stream_id,
                        "neuros_window_id": item.window_id,
                        "window_start_time_ns": item.start_time_ns,
                        "window_end_time_ns": item.end_time_ns,
                        "window_sample_rate_hz": item.sample_rate_hz,
                        "window_channel_names": item.channel_names,
                        "source_sequence_ids": item.source_sequence_ids,
                        "window_quality": int(item.quality),
                    },
                )
            return result
        if node.kind is NodeKind.SINK:
            if not hasattr(operator, "write"):
                raise TypeError(f"Sink node {node.node_id} lacks write()")
            await self._invoke_callable(node, operator.write, item)
            return None
        raise ProcessingError(
            f"Unsupported unary node kind: {node.kind.value}"
        )

    async def _invoke_callable(
        self, node: RuntimeNode, func: Callable[[Any], Any], item: Any
    ) -> Any:
        execution = ExecutionClass(node.executor)
        if execution in (ExecutionClass.INLINE, ExecutionClass.GPU):
            result = func(item)
            if inspect.isawaitable(result):
                return await result
            return result
        if execution is ExecutionClass.THREAD:
            return await asyncio.to_thread(_call, func, item)
        if execution is ExecutionClass.PROCESS:
            timeout_s = node.execution_timeout_s
            if timeout_s is None:
                raise ValueError(
                    f"Process node {node.node_id} lacks execution_timeout_s"
                )
            method_name = getattr(func, "__name__", None)
            if (
                not isinstance(method_name, str)
                or not method_name
                or getattr(func, "__self__", None) is not node.operator
            ):
                raise TypeError(
                    f"Process node {node.node_id} requires a bound operator method"
                )
            worker = self._process_workers.get(node.node_id)
            if worker is None:
                if node.process_transport == "shared_memory":
                    request_capacity = node.process_request_capacity_bytes
                    response_capacity = node.process_response_capacity_bytes
                    if request_capacity is None or response_capacity is None:
                        raise ValueError(
                            f"Shared-memory process node {node.node_id} lacks "
                            "explicit request/response mailbox capacities"
                        )
                    worker = SharedMemoryProcessWorker(
                        node.node_id,
                        node.operator,
                        execution_timeout_s=timeout_s,
                        request_capacity_bytes=request_capacity,
                        response_capacity_bytes=response_capacity,
                    )
                else:
                    worker = PersistentProcessWorker(
                        node.node_id,
                        node.operator,
                        execution_timeout_s=timeout_s,
                    )
                self._process_workers[node.node_id] = worker
            receipts = self._process_receipts.setdefault(
                node.node_id, deque(maxlen=4096)
            )
            try:
                call = await worker.invoke(method_name, item)
                return call.result
            finally:
                receipt = worker.last_receipt
                if receipt is not None and (
                    not receipts
                    or receipts[-1].get("request_id") != receipt.request_id
                    or receipts[-1].get("generation") != receipt.generation
                ):
                    receipts.append(receipt.as_dict())
        raise ValueError(f"Unsupported execution class: {execution.value}")

    async def _emit(self, node: RuntimeNode, item: Any) -> None:
        await self._notify_monitors(node, item)
        if isinstance(item, DecoderOutput) or node.kind is NodeKind.DECODER:
            await self._output_queue.put(item)
        outgoing = self._outgoing[node.node_id]
        if not outgoing:
            return
        for channel in outgoing:
            await put_with_policy(
                channel.queue,
                item,
                policy=channel.policy,
                stats=channel.stats,
            )

    async def _emit_stop(self, node_id: str) -> None:
        for channel in self._outgoing[node_id]:
            # Shutdown markers are control-plane authority and must never be
            # dropped by the data overflow policy.
            await channel.queue.put(_STOP)

    async def _notify_monitors(self, node: RuntimeNode, item: Any) -> None:
        for monitor_node in self.graph.nodes.values():
            if monitor_node.kind is not NodeKind.MONITOR:
                continue
            monitor = monitor_node.operator
            if not hasattr(monitor, "update"):
                continue
            payload = {
                "node_id": node.node_id,
                "kind": node.kind.value,
                "item": item,
                "monotonic_time_ns": time.monotonic_ns(),
            }
            started = time.perf_counter_ns()
            try:
                await self._invoke_callable(
                    monitor_node, monitor.update, payload
                )
                self._node_stats[monitor_node.node_id].observe(
                    time.perf_counter_ns() - started
                )
            except Exception as exc:
                raise _AttributedNodeError(monitor_node.node_id, exc) from exc

    async def outputs(self) -> AsyncIterator[Any]:
        """Subscribe to decoder outputs until the runtime terminates."""
        if self.state is RuntimeState.CREATED:
            raise RuntimeError("start() must be called before outputs()")
        while True:
            if (
                self._completion_task is not None
                and self._completion_task.done()
                and self._output_queue.empty()
            ):
                return
            try:
                item = await asyncio.wait_for(
                    self._output_queue.get(), timeout=0.05
                )
            except asyncio.TimeoutError:
                continue
            self._output_queue.task_done()
            yield item

    async def events(self) -> AsyncIterator[RuntimeEvent]:
        while True:
            if (
                self._completion_task is not None
                and self._completion_task.done()
                and self._event_queue.empty()
            ):
                return
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=0.05
                )
            except asyncio.TimeoutError:
                continue
            self._event_queue.task_done()
            yield event

    async def wait(self) -> None:
        if self._completion_task is None:
            raise RuntimeError("Runtime has not been started")
        await self._completion_task
        if self.failure is not None:
            raise RuntimeError(
                f"Runtime failed at {self.failure.node_id}: "
                f"{self.failure.error_type}: {self.failure.message}"
            )

    async def stop(self) -> None:
        if self.state is RuntimeState.STOPPED:
            return
        if self.state is RuntimeState.FAILED:
            # FAILED is scientific terminal authority, but callers invoking
            # stop() also receive a resource-cleanup barrier.
            if self._completion_task is not None and not self._completion_task.done():
                await asyncio.gather(
                    self._completion_task, return_exceptions=True
                )
            await self._close_process_workers()
            return
        if self.state is RuntimeState.CREATED:
            await self._close_process_workers()
            if self.failure is None:
                self.stopped_ns = time.monotonic_ns()
                await self._transition(RuntimeState.STOPPED, "runtime_stopped")
            return

        self._stopping = True
        if self.state is not RuntimeState.DRAINING:
            await self._transition(RuntimeState.DRAINING, "runtime_draining")
        source_tasks = [
            self._tasks[node_id]
            for node_id, node in self.graph.nodes.items()
            if node.kind is NodeKind.SOURCE and node_id in self._tasks
        ]
        for task in source_tasks:
            if not task.done():
                task.cancel()

        try:
            if self._completion_task is not None:
                await asyncio.wait_for(
                    asyncio.shield(self._completion_task),
                    timeout=self.drain_timeout_s,
                )
        except asyncio.TimeoutError:
            pending_node_ids = tuple(
                sorted(
                    node_id
                    for node_id, task in self._tasks.items()
                    if not task.done()
                )
            )
            timeout_error = RuntimeDrainTimeoutError(
                self.drain_timeout_s, pending_node_ids
            )
            if self.failure is None:
                self.failure = RuntimeFailure(
                    "runtime",
                    type(timeout_error).__name__,
                    str(timeout_error),
                )
                self.stopped_ns = time.monotonic_ns()
                await self._transition(
                    RuntimeState.FAILED,
                    "runtime_drain_timeout",
                    drain_timeout_s=self.drain_timeout_s,
                    pending_node_ids=pending_node_ids,
                    error_type=type(timeout_error).__name__,
                    message=str(timeout_error),
                )
            for task in self._tasks.values():
                if not task.done():
                    task.cancel()
            if self._completion_task is not None:
                await asyncio.gather(
                    self._completion_task, return_exceptions=True
                )
        finally:
            if self._completion_task is not None and self._completion_task.done():
                await self._close_process_workers()

        if self.state is not RuntimeState.FAILED:
            await self._finish_successfully()

    async def run(self) -> dict[str, Any]:
        await self.start()
        await self.wait()
        return self.snapshot()

    async def run_for(self, duration_s: float) -> dict[str, Any]:
        """Run until the duration expires or the graph terminates first.

        Earlier implementations slept for the full requested duration even when
        a node had already failed, then returned a failed snapshot without
        raising. Waiting on the completion task and timer concurrently preserves
        the same failure semantics as :meth:`run` and :meth:`wait`.
        """

        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        await self.start()
        if self._completion_task is None:
            raise RuntimeError("Runtime completion authority was not created")

        timer = asyncio.create_task(
            asyncio.sleep(duration_s), name="neuros:duration"
        )
        done, _ = await asyncio.wait(
            {timer, self._completion_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if self._completion_task in done:
            timer.cancel()
            await asyncio.gather(timer, return_exceptions=True)
            await self.wait()
            return self.snapshot()

        await self.stop()
        await self.wait()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        edge_metrics = {
            f"{source}->{target}": {
                "accepted": channel.stats.accepted,
                "dropped": channel.stats.dropped,
                "high_water_mark": channel.stats.high_water_mark,
                "capacity": channel.edge.capacity,
                "overflow_policy": channel.policy.value,
            }
            for (source, target), channel in self._channels.items()
        }
        runtime_ns = 0
        if self.started_ns is not None:
            runtime_ns = (
                self.stopped_ns or time.monotonic_ns()
            ) - self.started_ns
        return {
            "state": self.state.value,
            "runtime_seconds": runtime_ns / 1_000_000_000.0,
            "failure": None
            if self.failure is None
            else {
                "node_id": self.failure.node_id,
                "error_type": self.failure.error_type,
                "message": self.failure.message,
            },
            "nodes": {
                node_id: stats.snapshot()
                for node_id, stats in self._node_stats.items()
            },
            "edges": edge_metrics,
            "process_receipts": {
                node_id: list(receipts)
                for node_id, receipts in self._process_receipts.items()
            },
        }
