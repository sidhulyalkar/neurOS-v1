import os
from fastapi.testclient import TestClient

from neuros.api.server import app


def test_api_token_protection(monkeypatch):
    # set plain API token
    monkeypatch.setenv("NEUROS_API_TOKEN", "secret")
    client = TestClient(app)
    # missing Authorization header should fail
    resp = client.post("/train", json={"features": [], "labels": []})
    assert resp.status_code == 401
    # correct token should succeed (but request invalid due to mismatched data)
    headers = {"Authorization": "Bearer secret"}
    resp = client.post("/train", json={"features": [], "labels": []}, headers=headers)
    # this should fail with 400 but not 401
    assert resp.status_code != 401

    # now test hashed token authentication: compute hash of token
    import hashlib
    digest = hashlib.sha256(b"secret").hexdigest()
    monkeypatch.setenv("NEUROS_API_TOKEN", "")
    monkeypatch.setenv("NEUROS_API_TOKEN_HASH", digest)
    client_hash = TestClient(app)
    # missing header should fail
    resp2 = client_hash.post("/train", json={"features": [], "labels": []})
    assert resp2.status_code == 401
    # correct hashed token should authenticate
    headers2 = {"Authorization": "Bearer secret"}
    resp3 = client_hash.post("/train", json={"features": [], "labels": []}, headers=headers2)
    assert resp3.status_code != 401