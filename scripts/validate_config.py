#!/usr/bin/env python3
"""Validate a neurOS YAML configuration and optionally resolve its plugins."""

from __future__ import annotations

import argparse
import json

from neuros.config import load_config, resolve_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a neurOS YAML configuration")
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Validate schema without instantiating installed plugins",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    summary = {
        "schema_version": config.schema_version,
        "streams": [stream.stream_id for stream in config.streams],
        "decoder": config.decoder.plugin,
        "queue_capacity": config.runtime.queue_capacity,
        "overflow_policy": config.runtime.overflow_policy.value,
    }

    if not args.schema_only:
        resolved = resolve_config(config)
        summary["graph_nodes"] = sorted(resolved.graph.nodes)
        summary["graph_edges"] = [
            f"{edge.source}->{edge.target}" for edge in resolved.graph.edges
        ]

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
