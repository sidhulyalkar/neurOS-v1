"""
Streamlit dashboard for neurOS.

This dashboard demonstrates how to visualise real‑time neural data, feature
traces and classification outputs.  It requires the optional `streamlit`
dependency.
"""

from __future__ import annotations

import asyncio
import time
from typing import Deque, Tuple

import numpy as np

try:
    import streamlit as st  # type: ignore
except ImportError:  # pragma: no cover
    st = None

# NOTE: Use absolute imports so that this module can be executed as a
# standalone script via ``streamlit run``.  When run from within the
# installed ``neuros`` package the ``neuros`` package is on ``sys.path``,
# so these imports resolve correctly.  When executed as a script
# (e.g. by Streamlit) there is no package context and relative imports
# fail, therefore absolute imports are required.
from neuros.drivers.mock_driver import MockDriver
from neuros.models.simple_classifier import SimpleClassifier
from neuros.pipeline import Pipeline
from neuros.db.database import Database
import pandas as pd  # type: ignore
import os


def launch_dashboard() -> None:
    """Launch an analytics dashboard for inspecting neurOS runs and streams.

    The dashboard displays an overview of all recorded runs with average
    throughput and latency, allows comparison of metrics across runs and
    provides drill‑downs into individual runs.  Cross‑run comparisons
    and per‑run result plots enable users to evaluate performance and
    diagnose issues.
    """
    if st is None:
        raise RuntimeError("streamlit is not installed")

    # configure page layout and title
    st.set_page_config(page_title="neurOS Analytics Dashboard", layout="wide")
    st.title("neurOS Analytics Dashboard")
    # connect to the database
    db_path = os.getenv("NEUROS_DB_PATH", "neuros.db")
    db = Database(db_path)
    # choose view from sidebar
    view = st.sidebar.selectbox(
        "Select dashboard view",
        ["Overview", "Run details", "Custom pipeline demo"],
        index=0,
    )
    # list all runs (tenant filtering can be added by setting NEUROS_TENANT_ID)
    tenant_filter = os.getenv("NEUROS_TENANT_ID")
    run_ids = db.list_runs(tenant_id=tenant_filter) if tenant_filter else db.list_runs()
    if not run_ids:
        st.write("No runs have been recorded yet. Use the API to start pipelines.")
        return

    # load metrics for all runs
    metrics_list = []
    for rid in run_ids:
        m = db.get_run_metrics(rid, tenant_id=tenant_filter) if tenant_filter else db.get_run_metrics(rid)
        if m:
            metrics_list.append(m)
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        if view == "Overview":
            st.subheader("Overview of Runs")
            # allow filtering by driver and model for cross‑modal analysis
            drivers = [d for d in metrics_df.get("driver", []).unique() if d]
            models = [m for m in metrics_df.get("model", []).unique() if m]
            # handle None values
            if drivers:
                driver_filter = st.selectbox("Filter by driver", ["All"] + sorted(drivers))
                if driver_filter != "All":
                    metrics_df = metrics_df[metrics_df["driver"] == driver_filter]
            if models:
                model_filter = st.selectbox("Filter by model", ["All"] + sorted(models))
                if model_filter != "All":
                    metrics_df = metrics_df[metrics_df["model"] == model_filter]
            # show metrics table with sorting
            st.dataframe(metrics_df, use_container_width=True)
            # display aggregate statistics for filtered runs
            avg_throughput = metrics_df["throughput"].mean()
            avg_latency = metrics_df["mean_latency"].mean()
            st.metric("Average throughput (samples/s)", f"{avg_throughput:.2f}")
            st.metric("Average latency (s)", f"{avg_latency:.4f}")
            # show average quality metrics if available
            if "quality_mean" in metrics_df.columns and metrics_df["quality_mean"].notnull().any():
                avg_q_mean = metrics_df["quality_mean"].mean()
                avg_q_std = metrics_df["quality_std"].mean()
                st.metric("Average signal mean", f"{avg_q_mean:.4f}")
                st.metric("Average signal std", f"{avg_q_std:.4f}")
            # cross‑run comparison: select metric and show bar chart
            metric_to_plot = st.selectbox(
                "Select metric for comparison",
                [
                    "throughput",
                    "mean_latency",
                    "duration",
                    "samples",
                    "accuracy",
                    "quality_mean",
                    "quality_std",
                ],
            )
            # only plot selected metric if exists in dataframe
            if metric_to_plot in metrics_df.columns:
                st.bar_chart(metrics_df.set_index("run_id")[metric_to_plot])

    if view == "Run details":
        # allow user to select one or more runs to examine
        selected_runs = st.multiselect("Select run(s) for detail", run_ids, default=run_ids[:1])
        for run_id in selected_runs:
            metrics = (
                db.get_run_metrics(run_id, tenant_id=tenant_filter)
                if tenant_filter
                else db.get_run_metrics(run_id)
            )
            if not metrics:
                continue
            st.subheader(f"Run {run_id} Metrics")
            # display metrics in a table for readability
            m_df = pd.DataFrame.from_dict(
                {k: [v] for k, v in metrics.items()}, orient="columns"
            )
            st.table(m_df)
            # load results
            results = (
                db.get_stream_results(run_id, tenant_id=tenant_filter)
                if tenant_filter
                else db.get_stream_results(run_id)
            )
            if results:
                df = pd.DataFrame(results, columns=["timestamp", "label", "confidence", "latency"])
                # create tabs for different visualisations
                tabs = st.tabs(["Latency", "Label distribution", "Confidence"])
                with tabs[0]:
                    st.line_chart(df.set_index("timestamp")["latency"], height=200)
                with tabs[1]:
                    label_counts = df["label"].value_counts().sort_index()
                    st.bar_chart(label_counts)
                with tabs[2]:
                    st.line_chart(df.set_index("timestamp")["confidence"], height=200)
            # show quality metrics if available
            if metrics.get("quality_mean") is not None:
                st.info(
                    f"Signal quality: mean={metrics['quality_mean']:.4f}, std={metrics['quality_std']:.4f}"
                )

    elif view == "Custom pipeline demo":
        """
        Interactive demonstration of neurOS pipelines.  Users can select a driver and
        a model from predefined options, specify a run duration and execute a
        short pipeline to see the resulting metrics.  The demonstration uses
        random synthetic training data to initialise the model.  Additional
        drivers and models can be added here as the platform evolves.
        """
        st.subheader("Custom Pipeline Demo")
        # available drivers and models for demonstration
        drivers_dict = {
            "MockDriver": MockDriver,
        }
        models_dict = {
            "SimpleClassifier": SimpleClassifier,
        }
        driver_name = st.selectbox("Select driver", list(drivers_dict.keys()))
        model_name = st.selectbox("Select model", list(models_dict.keys()))
        duration_demo = st.slider("Run duration (seconds)", 1.0, 10.0, 3.0)
        if st.button("Run pipeline"):
            driver_cls = drivers_dict[driver_name]
            model_cls = models_dict[model_name]
            driver = driver_cls()
            model = model_cls()
            # train model if supported
            try:
                X_train = np.random.randn(100, 5 * driver.channels)
                y_train = np.random.randint(0, 2, size=100)
                if hasattr(model, "train"):
                    model.train(X_train, y_train)
            except Exception as exc:
                st.warning(f"Training failed: {exc}")
            # run pipeline and display metrics
            try:
                pipeline = Pipeline(driver=driver, model=model)
                metrics = asyncio.run(pipeline.run(duration=duration_demo))
                st.write("Run metrics:")
                st.json(metrics)
            except Exception as exc:
                st.error(f"Pipeline execution failed: {exc}")