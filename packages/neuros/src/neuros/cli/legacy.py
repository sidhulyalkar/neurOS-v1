"""Compatibility handlers for pre-v3 neurOS CLI commands."""

from __future__ import annotations

import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def handle(args: Any) -> bool:
    """Handle legacy commands. Return True when a command was consumed."""
    import neuros.cli as cli_api

    if args.command == "run" and args.config is None:
        from neuros.drivers.mock_driver import MockDriver
        from neuros.models.simple_classifier import SimpleClassifier

        pipeline = cli_api.Pipeline(driver=MockDriver(), model=SimpleClassifier())
        X_train = np.random.randn(100, 5 * pipeline.driver.channels)
        y_train = np.random.randint(0, 2, size=100)
        pipeline.train(X_train, y_train)
        metrics = cli_api.asyncio.run(pipeline.run(duration=args.duration))
        print(json.dumps(metrics, indent=2, default=str))
        return True

    if args.command == "benchmark" and args.config is None:
        metrics = cli_api.asyncio.run(cli_api.run_benchmark(duration=args.duration))
        text = json.dumps(metrics, indent=2, default=str)
        if args.report:
            Path(args.report).write_text(text, encoding="utf-8")
        else:
            print(text)
        return True

    if args.command == "train":
        import pandas as pd
        from neuros.models.simple_classifier import SimpleClassifier

        csv_path = Path(args.csv)
        df = pd.read_csv(csv_path)
        model = SimpleClassifier()
        model.train(df.iloc[:, :-1].values, df.iloc[:, -1].values)
        model_path = csv_path.with_suffix(".model.pkl")
        with model_path.open("wb") as handle_obj:
            pickle.dump(model, handle_obj)
        print(f"Model trained and saved to {model_path}")
        return True

    if args.command == "save-model":
        from neuros.models import ModelRegistry

        with open(args.model_file, "rb") as handle_obj:
            model = pickle.load(handle_obj)
        metrics = {"accuracy": args.accuracy} if args.accuracy is not None else {}
        metadata = ModelRegistry().save(
            model,
            name=args.name,
            version=args.version,
            metrics=metrics,
            tags=args.tags or [],
        )
        print(f"✓ Model saved: {metadata.name} v{metadata.version}")
        print(f"  Type: {metadata.model_type}")
        print(f"  Path: {metadata.file_path}")
        return True

    if args.command == "load-model":
        from neuros.models import ModelRegistry

        registry = ModelRegistry()
        model = registry.load(args.name, version=args.version)
        if args.output:
            with open(args.output, "wb") as handle_obj:
                pickle.dump(model, handle_obj)
            print(f"✓ Model loaded and saved to: {args.output}")
        else:
            version = args.version or registry.get_latest(args.name).version
            metadata = registry.get_metadata(args.name, version)
            print(f"✓ Model loaded: {metadata.name} v{metadata.version}")
            print(f"  Type: {metadata.model_type}")
            print(f"  Created: {metadata.created_at}")
        return True

    if args.command == "list-models":
        from neuros.models import ModelRegistry

        registry = ModelRegistry()
        if args.tags:
            models = registry.search(tags=args.tags)
        elif args.filter:
            models = registry.list_models(name_filter=args.filter)
        else:
            models = registry.list_models()
        if args.format == "json":
            print(json.dumps([item.to_dict() for item in models], indent=2, default=str))
        else:
            if not models:
                print("No models found in registry.")
                return True
            print(f"{'Name':<30} {'Version':<15} {'Type':<20} {'Created':<20} {'Accuracy':<10}")
            print("=" * 105)
            for item in models:
                accuracy = item.metrics.get("accuracy", "-")
                created = item.created_at[:19].replace("T", " ")
                print(f"{item.name:<30} {item.version:<15} {item.model_type:<20} {created:<20} {str(accuracy):<10}")
        return True

    if args.command == "dashboard":
        try:
            import streamlit  # noqa: F401
        except ImportError:
            print("streamlit is not installed; install the dashboard extra", file=sys.stderr)
            raise SystemExit(1)
        spec = importlib.util.find_spec("neuros.dashboard")
        if spec is None or not spec.origin:
            print("Unable to locate neuros.dashboard", file=sys.stderr)
            raise SystemExit(1)
        subprocess.run([sys.executable, "-m", "streamlit", "run", spec.origin], check=True)
        return True

    if args.command == "demo":
        from neuros.agents.notebook_agent import NotebookAgent

        path = NotebookAgent(output_dir=args.output_dir).generate_demo(
            args.task, duration=args.duration
        )
        print(f"Generated notebook: {path}")
        return True

    if args.command == "run-tasks":
        from neuros.agents.modality_manager_agent import ModalityManagerAgent

        agent = ModalityManagerAgent(args.tasks, duration=args.duration)
        print(json.dumps(cli_api.asyncio.run(agent.run_all()), indent=2, default=str))
        return True

    if args.command == "serve":
        try:
            import uvicorn
        except ImportError:
            print("uvicorn is not installed", file=sys.stderr)
            raise SystemExit(1)
        from neuros.api.server import app

        uvicorn.run(app, host=args.host, port=args.port)
        return True

    if args.command == "constellation":
        from neuros.cloud.pipeline_cloud import run_constellation_demo

        bootstrap = None if args.no_kafka else args.kafka_bootstrap
        cli_api.asyncio.run(
            run_constellation_demo(
                duration=args.duration,
                kafka_bootstrap=bootstrap,
                topic_prefix=args.topic_prefix,
                subject_id=args.subject_id,
                session_id=args.session_id,
                output_base=args.output_dir,
                fault_injection=args.fault_injection,
                sagemaker_config=args.sagemaker_config,
            )
        )
        return True

    return False
