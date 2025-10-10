import os
from fastapi.testclient import TestClient

from neuros.api.server import app


def test_api_token_protection(monkeypatch):
    # set plain API token
    monkeypatch.setenv("NEUROS_API_TOKEN", "secret")
    client = TestClient(app)

    # Valid training data for when auth succeeds
    valid_payload = {
        "features": [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
        "labels": [0, 1]
    }

    # missing Authorization header should fail with 401
    resp = client.post("/train", json=valid_payload)
    assert resp.status_code == 401

    # correct token should succeed
    headers = {"Authorization": "Bearer secret"}
    resp = client.post("/train", json=valid_payload, headers=headers)
    # should succeed (200) not fail with auth error (401)
    assert resp.status_code == 200
    assert resp.json()["status"] == "trained"

    # now test hashed token authentication: compute hash of token
    import hashlib
    digest = hashlib.sha256(b"secret").hexdigest()
    monkeypatch.setenv("NEUROS_API_TOKEN", "")
    monkeypatch.setenv("NEUROS_API_TOKEN_HASH", digest)
    client_hash = TestClient(app)

    # missing header should fail with 401
    resp2 = client_hash.post("/train", json=valid_payload)
    assert resp2.status_code == 401

    # correct hashed token should authenticate and succeed
    headers2 = {"Authorization": "Bearer secret"}
    resp3 = client_hash.post("/train", json=valid_payload, headers=headers2)
    assert resp3.status_code == 200
    assert resp3.json()["status"] == "trained"