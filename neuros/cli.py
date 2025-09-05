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

from .pipeline import Pipeline
from .drivers.mock_driver import MockDriver
from .models.simple_classifier import SimpleClassifier
from .benchmarks.benchmark_pipeline import run_benchmark


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
            from .agents.notebook_agent import NotebookAgent
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
            from .agents.modality_manager_agent import ModalityManagerAgent
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
        from .api.server import app

        uvicorn.run(app, host=args.host, port=args.port)

    elif args.command == "constellation":
        # Run the Constellation multi‑modal pipeline demo
        try:
            # Defer import so that optional dependencies are loaded only when needed
            from .cloud.pipeline_cloud import run_constellation_demo
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