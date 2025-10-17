"""
Label Studio and CVAT integration utilities.

Constellation relies on human‑in‑the‑loop annotation for behaviour and
pose labels.  This module provides helper functions to create
annotation tasks in Label Studio and CVAT, and to handle webhook
callbacks when an annotator completes a task.

For simplicity, this implementation uses the REST API documented at
https://labelstud.io/guide/api_reference.html and assumes you have
configured an API token and base URL via environment variables or
function arguments.  CVAT integration would follow a similar pattern
and is left as an exercise.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import os
import requests  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class LabelStudioConfig:
    """Configuration for Label Studio API."""

    base_url: str
    api_token: str


def create_label_studio_tasks(
    tasks: Iterable[Dict[str, any]],
    project_id: int,
    config: LabelStudioConfig,
) -> Dict[str, any]:
    """Create annotation tasks in Label Studio.

    Parameters
    ----------
    tasks:
        Iterable of task dictionaries.  Each task should contain a
        ``data`` field with the raw data to be annotated (e.g. a URL
        pointing to a video) and any metadata fields required by your
        Label Studio project.
    project_id:
        Identifier of the Label Studio project to attach tasks to.
    config:
        Connection configuration with base URL and API token.
    """
    headers = {
        "Authorization": f"Token {config.api_token}",
        "Content-Type": "application/json",
    }
    payload = {"tasks": list(tasks)}
    url = f"{config.base_url}/api/projects/{project_id}/tasks/bulk/"
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code != 201:
        logger.error(
            "Failed to create tasks in Label Studio: %s", response.text[:200]
        )
        raise RuntimeError("Failed to create Label Studio tasks")
    logger.info("Created %d tasks in Label Studio project %d", len(payload["tasks"]), project_id)
    return response.json()


def handle_label_studio_webhook(event: Dict[str, any]) -> None:
    """Process a Label Studio webhook event.

    This function should be registered as a webhook handler.  It
    extracts the annotation result and writes it into the lakehouse
    via your preferred ETL path (e.g. append to Iceberg ``labels.*``
    tables).  In this stub we simply log the event.
    """
    logger.info("Received Label Studio webhook: %s", json.dumps(event)[:200])
    # TODO: parse the event and store the annotation into labels.* tables


__all__ = ["LabelStudioConfig", "create_label_studio_tasks", "handle_label_studio_webhook"]