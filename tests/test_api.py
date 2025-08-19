import asyncio
import json
import pytest
from fastapi.testclient import TestClient

from neuros.api.server import app
from neuros.models.simple_classifier import SimpleClassifier


def _train_dummy_model(client: TestClient) -> None:
    # train a simple model via API using small dataset
    # features: 2 samples with 4 features each, labels binary
    payload = {
        "features": [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
        "labels": [0, 1],
    }
    response = client.post("/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "trained"
    assert data["samples"] == 2


def test_train_and_start_pipeline():
    client = TestClient(app)
    _train_dummy_model(client)
    # run pipeline for 1 second
    response = client.post("/start", json={"duration": 1.0})
    assert response.status_code == 200
    data = response.json()
    # check metrics keys
    assert "samples" in data
    assert "throughput" in data
    assert data["samples"] > 0
    assert data["throughput"] > 0
    assert "run_id" in data


def test_websocket_stream():
    client = TestClient(app)
    _train_dummy_model(client)
    # open websocket connection with small duration
    with client.websocket_connect("/stream?duration=1.0") as ws:
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