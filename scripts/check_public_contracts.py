#!/usr/bin/env python3
"""Fail closed when public trust/release contracts drift out of the repository."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "SUPPORT.md",
    "GOVERNANCE.md",
    "CHANGELOG.md",
    "docs/SCIENTIFIC_CLAIMS.md",
    "docs/RELEASE_POLICY.md",
)


def fail(message: str) -> None:
    raise SystemExit(f"public-contract check failed: {message}")


def read(path: str) -> str:
    file_path = ROOT / path
    if not file_path.is_file():
        fail(f"missing required file {path}")
    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
        fail(f"required file {path} is empty")
    return text


def validate_citation() -> None:
    payload = yaml.safe_load(read("CITATION.cff"))
    if not isinstance(payload, dict):
        fail("CITATION.cff must decode to a mapping")
    if str(payload.get("cff-version")) != "1.2.0":
        fail("CITATION.cff must use cff-version 1.2.0")
    for key in ("message", "title", "type", "authors", "repository-code", "license"):
        if not payload.get(key):
            fail(f"CITATION.cff missing {key}")
    if payload.get("license") != "MIT":
        fail("CITATION.cff license must match repository MIT license")
    authors = payload.get("authors")
    if not isinstance(authors, list) or not authors or not all(isinstance(a, dict) for a in authors):
        fail("CITATION.cff authors must be a non-empty list of mappings")
    repository = str(payload.get("repository-code", ""))
    if repository.rstrip("/") != "https://github.com/sidhulyalkar/neurOS-v1":
        fail("CITATION.cff repository-code must identify the canonical repository")
    # Do not allow a placeholder/fabricated DOI to quietly become citation authority.
    doi = payload.get("doi")
    if doi is not None and not re.fullmatch(r"10\.\d{4,9}/\S+", str(doi)):
        fail("CITATION.cff DOI is malformed; omit DOI until a real archival DOI exists")


def validate_claim_ladder() -> None:
    claims = read("docs/SCIENTIFIC_CLAIMS.md").lower()
    required_tiers = (
        "software contract",
        "integration",
        "real dataset",
        "hardware",
        "closed loop",
        "clinical",
    )
    missing = [tier for tier in required_tiers if tier not in claims]
    if missing:
        fail(f"scientific claim policy missing evidence tiers: {missing}")
    for statement in (
        "hardware qualification does not imply closed-loop qualification",
        "closed-loop qualification does not imply medical-device certification",
        "if evidence and wording disagree, the evidence tier wins",
    ):
        if statement not in claims:
            fail(f"scientific claim policy lost required fail-closed statement: {statement!r}")


def validate_release_policy() -> None:
    policy = read("docs/RELEASE_POLICY.md").lower()
    required = (
        "sha-256",
        "exact release commit",
        "trusted publishing",
        "pull-request ci must not possess package-publishing credentials",
        "twine check",
    )
    missing = [term for term in required if term not in policy]
    if missing:
        fail(f"release policy missing required release controls: {missing}")


def validate_docs_navigation() -> None:
    mkdocs = read("mkdocs.yml")
    for path in ("SCIENTIFIC_CLAIMS.md", "RELEASE_POLICY.md"):
        if path not in mkdocs:
            fail(f"mkdocs navigation does not expose {path}")


def main() -> None:
    for path in REQUIRED_FILES:
        read(path)
    validate_citation()
    validate_claim_ladder()
    validate_release_policy()
    validate_docs_navigation()
    print("public trust contracts: PASS")


if __name__ == "__main__":
    main()
