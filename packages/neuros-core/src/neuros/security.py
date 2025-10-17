"""
Security utilities for neurOS.

This module implements token hashing, loading of API key maps and
role‑based access control (RBAC).  It supports hashed tokens to avoid
storing secrets in plain text and allows associating tokens with
roles and tenants for multi‑tenant deployments.

Environment Variables
---------------------
NEUROS_API_TOKEN : str (optional)
    Plaintext token for backward compatibility.  If set, any request
    bearing this token will be authorised as an admin.
NEUROS_API_TOKEN_HASH : str (optional)
    SHA‑256 hash of a token.  Requests bearing a token whose hash
    matches will be authorised as an admin.
NEUROS_API_KEYS_JSON : str (optional)
    JSON string mapping token hashes to role and tenant information,
    e.g. ``{"<hash>": {"role": "viewer", "tenant": "org1"}}``.  If
    provided, the system will use this map for RBAC.

The default role is ``admin`` when no key map is configured.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Optional


def hash_token(token: str) -> str:
    """Compute the SHA‑256 hash of a token string."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def load_token_map() -> Dict[str, Dict[str, str]]:
    """Load the API key map from the environment.

    Returns
    -------
    dict
        Mapping of token hashes to dictionaries with ``role`` and
        ``tenant`` entries.
    """
    data = os.getenv("NEUROS_API_KEYS_JSON")
    if not data:
        return {}
    try:
        obj = json.loads(data)
        # ensure proper structure
        if not isinstance(obj, dict):
            raise ValueError
        for k, v in obj.items():
            if not isinstance(v, dict) or "role" not in v:
                raise ValueError
        return obj
    except Exception:
        raise RuntimeError("Invalid NEUROS_API_KEYS_JSON; must be JSON mapping token hashes to role info")


def get_token_info(token: str) -> Dict[str, str]:
    """Retrieve role and tenant information for a token.

    If no token map is configured, returns a default admin role.  If the
    token matches the legacy plain token or hash variables, returns admin.
    Otherwise looks up the SHA‑256 hash in the token map.
    """
    # legacy token/hashing
    plain = os.getenv("NEUROS_API_TOKEN")
    hsh = os.getenv("NEUROS_API_TOKEN_HASH")
    if plain and token == plain:
        return {"role": "admin", "tenant": "default"}
    if hsh and hash_token(token) == hsh:
        return {"role": "admin", "tenant": "default"}
    # token map
    token_map = load_token_map()
    if not token_map:
        # no map means all tokens are admin
        return {"role": "admin", "tenant": "default"}
    digest = hash_token(token)
    info = token_map.get(digest)
    if info is None:
        raise PermissionError("Invalid token")
    # ensure tenant field
    role = info.get("role", "viewer")
    tenant = info.get("tenant", "default")
    return {"role": role, "tenant": tenant}


def require_role(required_roles: str | tuple[str, ...]):
    """FastAPI dependency factory enforcing role requirements.

    Returns a dependency callable that extracts the bearer token from
    the Authorization header, resolves its role via the token map and
    verifies it matches one of the required roles.  If no token map is
    configured, all requests are permitted as admin.  Raises an
    HTTPException on failure.

    Parameters
    ----------
    required_roles : str or tuple[str, ...]
        The role or roles allowed to access the endpoint.
    """
    from fastapi import HTTPException, Header

    if isinstance(required_roles, str):
        allowed = (required_roles,)
    else:
        allowed = required_roles

    async def dependency(authorization: Optional[str] = Header(default=None)) -> Dict[str, str]:
        # extract token
        if authorization is None or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = authorization[len("Bearer "):].strip()
        try:
            info = get_token_info(token)
        except PermissionError:
            raise HTTPException(status_code=401, detail="Invalid token")
        if info["role"] not in allowed:
            raise HTTPException(status_code=403, detail="Forbidden")
        return info

    return dependency