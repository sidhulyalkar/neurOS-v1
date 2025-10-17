"""
Command line interface for neurOS.

This module defines the ``neuros`` entry point.  It provides commands to
run pipelines, train models, benchmark performance and launch the
dashboard.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from neuros.pipeline import Pipeline
from neuros.drivers.mock_driver import MockDriver
from neuros.models.simple_classifier import SimpleClassifier
from neuros.benchmarks.benchmark_pipeline import run_benchmark


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="neuros", description="neurOS CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run a real‑time pipeline")
    run_parser.add_argument("--duration", type=float, default=5.0, help="Run duration in seconds")

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--duration", type=float, default=10.0, help="Benchmark duration in seconds")
    bench_parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to write JSON report (prints to stdout if omitted)",
    )

    # train command
    train_parser = subparsers.add_parser("train", help="Train a model on CSV data")
    train_parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with features and labels")

    # save-model command
    save_model_parser = subparsers.add_parser("save-model", help="Save a trained model to the registry")
    save_model_parser.add_argument("--model-file", type=str, required=True, help="Path to pickled model file")
    save_model_parser.add_argument("--name", type=str, required=True, help="Model name")
    save_model_parser.add_argument("--version", type=str, help="Version (default: auto-generated timestamp)")
    save_model_parser.add_argument("--tags", nargs="+", help="Tags for organization")
    save_model_parser.add_argument("--accuracy", type=float, help="Model accuracy")

    # load-model command
    load_model_parser = subparsers.add_parser("load-model", help="Load a model from the registry")
    load_model_parser.add_argument("--name", type=str, required=True, help="Model name")
    load_model_parser.add_argument("--version", type=str, help="Version (default: latest)")
    load_model_parser.add_argument("--output", type=str, help="Output path for loaded model")

    # list-models command
    list_models_parser = subparsers.add_parser("list-models", help="List all models in the registry")
    list_models_parser.add_argument("--filter", type=str, help="Filter by name (substring match)")
    list_models_parser.add_argument("--tags", nargs="+", help="Filter by tags")
    list_models_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")

    # dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Launch the Streamlit dashboard")

    # demo command
    demo_parser = subparsers.add_parser("demo", help="Generate a Jupyter notebook demonstration for a task")
    demo_parser.add_argument("--task", type=str, required=True, help="Task description for the demo (e.g. '2-class motor imagery')")
    demo_parser.add_argument(
        "--duration", type=float, default=3.0, help="Duration in seconds to run the pipeline in the notebook"
    )
    demo_parser.add_argument(
        "--output-dir",
        type=str,
        default="notebooks",
        help="Directory to save the generated notebook (default: notebooks)",
    )

    # run tasks command
    run_tasks_parser = subparsers.add_parser(
        "run-tasks",
        help="Run pipelines for multiple task descriptions and return metrics",
    )
    run_tasks_parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="List of task descriptions to evaluate",
    )
    run_tasks_parser.add_argument(
        "--duration", type=float, default=3.0, help="Duration in seconds for each pipeline run"
    )

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Launch the FastAPI server")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")

    # constellation command
    const_parser = subparsers.add_parser(
        "constellation",
        help="Run the Constellation demo pipeline with multi‑modal ingestion, storage, export and training",
    )
    const_parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Ingestion duration in seconds (per modality)",
    )
    const_parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/constellation_demo",
        help="Base directory for raw and processed data (simulating S3)",
    )
    const_parser.add_argument(
        "--subject-id",
        type=str,
        default="demo_subject",
        help="Subject identifier used in metadata and file names",
    )
    const_parser.add_argument(
        "--session-id",
        type=str,
        default="demo_session",
        help="Session identifier used in metadata and file names",
    )
    const_parser.add_argument(
        "--fault-injection",
        action="store_true",
        help="Enable synthetic fault injection (packet drops and jitter)",
    )
    const_parser.add_argument(
        "--sagemaker-config",
        type=str,
        default=None,
        help="Path to a JSON or YAML file specifying SageMaker job configuration",
    )
    const_parser.add_argument(
        "--kafka-bootstrap",
        type=str,
        default="localhost:9092",
        help="Kafka bootstrap servers (host:port)",
    )
    const_parser.add_argument(
        "--topic-prefix",
        type=str,
        default="raw",
        help="Prefix for Kafka topics (e.g. raw or aligned)",
    )

    const_parser.add_argument(
        "--no-kafka",
        action="store_true",
        help="Run the demo without a Kafka broker (use NoopWriter)",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "run":
        # run a simple pipeline with mock driver and model
        pipeline = Pipeline(driver=MockDriver(), model=SimpleClassifier())
        # train a trivial model on random data for demonstration
        X_train = np.random.randn(100, 5 * pipeline.driver.channels)
        y_train = np.random.randint(0, 2, size=100)
        pipeline.train(X_train, y_train)
        metrics = asyncio.run(pipeline.run(duration=args.duration))
        print(json.dumps(metrics, indent=2))

    elif args.command == "benchmark":
        metrics = asyncio.run(run_benchmark(duration=args.duration))
        if args.report:
            Path(args.report).write_text(json.dumps(metrics, indent=2))
        else:
            print(json.dumps(metrics, indent=2))

    elif args.command == "train":
        # training from CSV: expects last column to be label
        import pandas as pd

        csv_path = Path(args.csv)
        df = pd.read_csv(csv_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        model = SimpleClassifier()
        model.train(X, y)
        # save model using pickle
        model_path = csv_path.with_suffix(".model.pkl")
        import pickle

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model trained and saved to {model_path}")

    elif args.command == "save-model":
        # save a model to the registry with metadata
        from neuros.models import ModelRegistry
        import pickle

        # Load the model
        with open(args.model_file, "rb") as f:
            model = pickle.load(f)

        # Prepare metadata
        metrics = {}
        if args.accuracy is not None:
            metrics["accuracy"] = args.accuracy

        # Save to registry
        registry = ModelRegistry()
        metadata = registry.save(
            model,
            name=args.name,
            version=args.version,
            metrics=metrics,
            tags=args.tags or [],
        )

        print(f"✓ Model saved: {metadata.name} v{metadata.version}")
        print(f"  Type: {metadata.model_type}")
        print(f"  Path: {metadata.file_path}")
        if metrics:
            print(f"  Metrics: {json.dumps(metrics, indent=4)}")

    elif args.command == "load-model":
        # load a model from the registry
        from neuros.models import ModelRegistry
        import pickle

        registry = ModelRegistry()
        model = registry.load(args.name, version=args.version)

        if args.output:
            # Save to specified path
            with open(args.output, "wb") as f:
                pickle.dump(model, f)
            print(f"✓ Model loaded and saved to: {args.output}")
        else:
            # Just display info
            metadata = registry.get_metadata(args.name, args.version or registry.get_latest(args.name).version)
            print(f"✓ Model loaded: {metadata.name} v{metadata.version}")
            print(f"  Type: {metadata.model_type}")
            print(f"  Created: {metadata.created_at}")
            if metadata.metrics:
                print(f"  Metrics: {json.dumps(metadata.metrics, indent=4)}")

    elif args.command == "list-models":
        # list models in the registry
        from neuros.models import ModelRegistry

        registry = ModelRegistry()

        # Apply filters
        if args.tags:
            models = registry.search(tags=args.tags)
        elif args.filter:
            models = registry.list_models(name_filter=args.filter)
        else:
            models = registry.list_models()

        if not models:
            print("No models found in registry.")
            return

        if args.format == "json":
            # JSON output
            output = [m.to_dict() for m in models]
            print(json.dumps(output, indent=2))
        else:
            # Table output
            print(f"\n{'Name':<30} {'Version':<15} {'Type':<20} {'Created':<20} {'Accuracy':<10}")
            print("=" * 105)
            for m in models:
                accuracy = m.metrics.get("accuracy", "-")
                if isinstance(accuracy, float):
                    accuracy = f"{accuracy:.3f}"
                created = m.created_at[:19].replace("T", " ")  # Format timestamp
                print(f"{m.name:<30} {m.version:<15} {m.model_type:<20} {created:<20} {accuracy:<10}")
            print(f"\nTotal: {len(models)} models\n")

    elif args.command == "dashboard":
        """Launch the Streamlit dashboard in a proper script context.

        When run via ``neuros dashboard`` the Streamlit dashboard is started
        using ``streamlit run`` so that a valid ScriptRunContext is
        established.  This avoids the "missing ScriptRunContext" warning
        that occurs when calling Streamlit APIs directly from an ordinary
        Python process.  If Streamlit is not installed the user is
        informed accordingly.
        """
        try:
            import streamlit  # type: ignore  # noqa: F401
        except ImportError:
            print(
                "streamlit is not installed.  Install with `pip install streamlit` or include the 'dashboard' extra",
                file=sys.stderr,
            )
            sys.exit(1)

        # Use subprocess to invoke ``streamlit run`` with the path to the
        # dashboard module.  Streamlit does not support a ``-m`` option for
        # modules, so we resolve the file path via importlib.  Passing the
        # resolved path ensures that Streamlit creates a ScriptRunContext and
        # avoids errors such as "No such option: -m".  If Streamlit is not
        # installed or the dashboard cannot be found the user is informed.
        import subprocess
        import importlib.util
        try:
            spec = importlib.util.find_spec("neuros.dashboard")
            if spec is None or not spec.origin:
                print("Unable to locate the neuros.dashboard module.", file=sys.stderr)
                sys.exit(1)
            module_path = spec.origin
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", module_path],
                check=True,
            )
        except FileNotFoundError:
            print(
                "Unable to locate the 'streamlit' command. Please ensure Streamlit is installed and available on the PATH.",
                file=sys.stderr,
            )
            sys.exit(1)

    elif args.command == "demo":
        # generate a demonstration notebook for a task
        try:
            # import directly from the module to avoid circular dependencies
            from neuros.agents.notebook_agent import NotebookAgent
        except ImportError:
            print(
                "NotebookAgent could not be imported. Ensure neurOS is installed correctly.",
                file=sys.stderr,
            )
            sys.exit(1)
        agent = NotebookAgent(output_dir=args.output_dir)
        path = agent.generate_demo(args.task, duration=args.duration)
        print(f"Generated notebook: {path}")

    elif args.command == "run-tasks":
        # run multiple pipelines for specified tasks and return metrics
        try:
            # import directly from the module to avoid circular dependencies
            from neuros.agents.modality_manager_agent import ModalityManagerAgent
        except ImportError:
            print(
                "ModalityManagerAgent could not be imported. Ensure neurOS is installed correctly.",
                file=sys.stderr,
            )
            sys.exit(1)
        agent = ModalityManagerAgent(args.tasks, duration=args.duration)
        results = asyncio.run(agent.run_all())
        print(json.dumps(results, indent=2))

    elif args.command == "serve":
        # run the FastAPI server using uvicorn
        try:
            import uvicorn  # type: ignore
        except ImportError:
            print(
                "uvicorn is not installed.  Install with `pip install uvicorn` or include it in requirements.",
                file=sys.stderr,
            )
            sys.exit(1)
        # import app lazily to allow optional FastAPI installation
        from neuros.api.server import app

        uvicorn.run(app, host=args.host, port=args.port)

    elif args.command == "constellation":
        # Run the Constellation multi‑modal pipeline demo
        try:
            # Defer import so that optional dependencies are loaded only when needed
            from neuros.cloud.pipeline_cloud import run_constellation_demo
        except Exception as exc:
            print(
                "Constellation pipeline is not available. Ensure optional dependencies are installed and the neuros.cloud package is present.",
                file=sys.stderr,
            )
            # print original error for debugging when run with verbose output
            # Commented out to avoid leaking stack traces by default
            # print(f"Import error: {exc}", file=sys.stderr)
            sys.exit(1)
        # run the async demo
        # Override Kafka bootstrap with None if the user requested no Kafka
        kafka_bootstrap = None if getattr(args, "no_kafka", False) else args.kafka_bootstrap
        asyncio.run(
            run_constellation_demo(
                duration=args.duration,
                kafka_bootstrap=kafka_bootstrap,
                topic_prefix=args.topic_prefix,
                subject_id=args.subject_id,
                session_id=args.session_id,
                output_base=args.output_dir,
                fault_injection=args.fault_injection,
                sagemaker_config=args.sagemaker_config,
            )
        )



if __name__ == "__main__":
    main()
