from __future__ import annotations

import numpy as np
import pytest

from neuros.runtime import NodeKind, RuntimeExecutor, RuntimeGraph, RuntimeNode
from neuros.runtime.process_worker import PersistentProcessWorker
from neuros.runtime.shared_process_worker import SharedMemoryProcessWorker


class IdentityTransform:
    def transform(self, item):
        return item


INVALID_DURATION_VALUES = (
    True,
    np.bool_(True),
    "1.0",
    float("nan"),
    float("inf"),
    float("-inf"),
    0,
    0.0,
    -1,
    -0.5,
)


@pytest.mark.parametrize("value", INVALID_DURATION_VALUES)
def test_runtime_node_rejects_invalid_execution_timeout_authority(value):
    with pytest.raises((TypeError, ValueError)):
        RuntimeNode(
            "transform",
            NodeKind.TRANSFORM,
            IdentityTransform(),
            executor="process",
            execution_timeout_s=value,
        )


@pytest.mark.parametrize("value", INVALID_DURATION_VALUES)
def test_runtime_node_rejects_invalid_latency_budget(value):
    with pytest.raises((TypeError, ValueError)):
        RuntimeNode(
            "transform",
            NodeKind.TRANSFORM,
            IdentityTransform(),
            latency_budget_ms=value,
        )


@pytest.mark.parametrize("value", INVALID_DURATION_VALUES)
def test_executor_rejects_invalid_drain_timeout_authority(value):
    with pytest.raises((TypeError, ValueError)):
        RuntimeExecutor(RuntimeGraph(), drain_timeout_s=value)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", INVALID_DURATION_VALUES)
async def test_run_for_rejects_invalid_duration_before_starting_runtime(value):
    executor = RuntimeExecutor(RuntimeGraph())
    with pytest.raises((TypeError, ValueError)):
        await executor.run_for(value)
    assert executor.state.value == "created"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("execution_timeout_s", value)
        for value in INVALID_DURATION_VALUES
    ]
    + [
        ("startup_timeout_s", value)
        for value in INVALID_DURATION_VALUES
    ]
    + [
        ("termination_grace_s", value)
        for value in INVALID_DURATION_VALUES
    ],
)
def test_pickle_worker_rejects_invalid_lifecycle_duration_authority(field_name, value):
    kwargs = {
        "execution_timeout_s": 1.0,
        "startup_timeout_s": 1.0,
        "termination_grace_s": 0.25,
    }
    kwargs[field_name] = value
    with pytest.raises((TypeError, ValueError)):
        PersistentProcessWorker("transform", IdentityTransform(), **kwargs)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("execution_timeout_s", value)
        for value in INVALID_DURATION_VALUES
    ]
    + [
        ("startup_timeout_s", value)
        for value in INVALID_DURATION_VALUES
    ]
    + [
        ("termination_grace_s", value)
        for value in INVALID_DURATION_VALUES
    ],
)
def test_shared_worker_rejects_invalid_lifecycle_duration_authority(field_name, value):
    kwargs = {
        "execution_timeout_s": 1.0,
        "startup_timeout_s": 1.0,
        "termination_grace_s": 0.25,
        "request_capacity_bytes": 4096,
        "response_capacity_bytes": 4096,
    }
    kwargs[field_name] = value
    with pytest.raises((TypeError, ValueError)):
        SharedMemoryProcessWorker("transform", IdentityTransform(), **kwargs)


def test_valid_real_duration_inputs_are_canonicalized_without_transport_drift():
    node = RuntimeNode(
        "transform",
        NodeKind.TRANSFORM,
        IdentityTransform(),
        executor="process",
        latency_budget_ms=np.float32(12.5),
        execution_timeout_s=np.float64(1.25),
    )
    executor = RuntimeExecutor(RuntimeGraph(), drain_timeout_s=np.float32(2.5))
    pickle_worker = PersistentProcessWorker(
        "pickle",
        IdentityTransform(),
        execution_timeout_s=np.float32(1.5),
        startup_timeout_s=np.float64(2.0),
        termination_grace_s=np.int64(1),
    )
    shared_worker = SharedMemoryProcessWorker(
        "shared",
        IdentityTransform(),
        execution_timeout_s=np.float64(1.75),
        startup_timeout_s=np.float32(2.25),
        termination_grace_s=np.int64(1),
        request_capacity_bytes=4096,
        response_capacity_bytes=4096,
    )

    assert node.latency_budget_ms == 12.5
    assert type(node.latency_budget_ms) is float
    assert node.execution_timeout_s == 1.25
    assert type(node.execution_timeout_s) is float
    assert executor.drain_timeout_s == 2.5
    assert type(executor.drain_timeout_s) is float
    assert pickle_worker.execution_timeout_s == 1.5
    assert pickle_worker.startup_timeout_s == 2.0
    assert pickle_worker.termination_grace_s == 1.0
    assert shared_worker.execution_timeout_s == 1.75
    assert shared_worker.startup_timeout_s == 2.25
    assert shared_worker.termination_grace_s == 1.0
