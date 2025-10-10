import asyncio
import json
import os
import pytest
from fastapi.testclient import TestClient

from neuros.api.server import app
from neuros.models.simple_classifier import SimpleClassifier


# Set test mode to disable authentication for tests
@pytest.fixture(autouse=True)
def setup_test_env():
    """Configure environment for testing - disable authentication."""
    # Set a test token that will be recognized
    os.environ["NEUROS_API_TOKEN"] = "test-token-123"
    yield
    # Clean up
    if "NEUROS_API_TOKEN" in os.environ:
        del os.environ["NEUROS_API_TOKEN"]


@pytest.fixture
def test_headers():
    """Headers with test authentication token."""
    return {"Authorization": "Bearer test-token-123"}


def _train_dummy_model(client: TestClient, headers: dict) -> None:
    # train a simple model via API using small dataset
    # The pipeline extracts 5 features per channel (default bands)
    # With 8 channels, that's 40 features total (8 * 5 = 40)
    # Create training data matching this dimension
    payload = {
        "features": [
            [0.1] * 40,  # 40 features for sample 1
            [0.2] * 40,  # 40 features for sample 2
            [0.3] * 40,  # 40 features for sample 3
            [0.4] * 40,  # 40 features for sample 4
        ],
        "labels": [0, 1, 0, 1],
    }
    response = client.post("/train", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "trained"
    assert data["samples"] == 4


def test_train_and_start_pipeline(test_headers):
    client = TestClient(app)
    _train_dummy_model(client, test_headers)
    # run pipeline for 1 second
    response = client.post("/start", json={"duration": 1.0}, headers=test_headers)
    assert response.status_code == 200
    data = response.json()
    # check metrics keys
    assert "samples" in data
    assert "throughput" in data
    assert data["samples"] > 0
    assert data["throughput"] > 0
    assert "run_id" in data


def test_websocket_stream(test_headers):
    client = TestClient(app)
    _train_dummy_model(client, test_headers)
    # open websocket connection with small duration
    # Pass authorization header via headers dict
    with client.websocket_connect(
        "/stream?duration=1.0",
        headers={"authorization": "Bearer test-token-123"}
    ) as ws:
        # receive a few messages
        count = 0
        for _ in range(3):
            msg = ws.receive_text()
            obj = json.loads(msg)
            assert "timestamp" in obj
            assert "label" in obj
            assert "confidence" in obj
            count += 1
        assert count == 3