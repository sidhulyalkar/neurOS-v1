# neurOS-v1 – A Modular Operating System for Brain‑Computer Interfaces

**neurOS-v1** is a reimagined version of the original neurOS codebase.  It focuses
on clarity, extensibility and measurable performance while delivering a
production‑ready platform for brain–computer interface (BCI) development and
research.  This repository contains a self‑contained Python package and
accompanying scripts for real‑time neural data streaming, processing and
classification.  It also includes benchmarking tools and a simple dashboard so
that researchers can evaluate the system end‑to‑end.

## Highlights

* **Streamlined drivers** – A unified API for reading data from real or simulated
  BCI hardware.  Drivers can be hot‑swapped without modifying the rest of the
  pipeline.
* **Pluggable processing pipeline** – Filters, feature extractors and models
  live in their own modules and are loaded dynamically at run time.  New
  algorithms can be added without changing core code.
* **Agent‑based orchestration** – An asynchronous orchestrator coordinates
  multiple agents (device, processing and model) to build a real‑time BCI
  pipeline.  Agents monitor performance and can adapt their behaviour when
  signals degrade.
* **Model training and inference** – Built‑in support for offline training and
  real‑time inference with state‑of‑the‑art neural and classical models.  Models
  implement a simple interface so that custom architectures can be added
  easily.
* **Benchmarking suite** – Scripts to measure latency, throughput and accuracy
  using synthetic and recorded datasets.  Benchmarks facilitate fair
  comparisons with other BCI frameworks.
* **Dashboard and CLI** – A lightweight Streamlit dashboard (optional) and
  command line interface make it easy to run pipelines, launch benchmarks and
  visualise data without writing code.

## Getting Started

### Installation

```bash
git clone https://github.com/shulyalk/neuros-v1.git
cd neuros-v1
pip install -r requirements.txt
pip install -e .

# run basic diagnostics
neuros --help
```

### Running a Pipeline

You can run a real‑time pipeline with a simulated driver using the CLI:

```bash
# run a pipeline that streams data for 5 seconds
neuros run --duration 5
```

The pipeline will print classification outputs to the terminal and record
latency metrics.  See `neuros run --help` for all options.

### Benchmarking

To evaluate the system performance on synthetic data, run:

```bash
neuros benchmark --duration 10 --report benchmarks/report.json
```

This command measures throughput, end‑to‑end latency and classification
accuracy over a configurable duration.  The resulting report can be used to
compare neurOS with other systems.

### Dashboard

An optional Streamlit dashboard is provided for interactive monitoring.  To
launch it, install `streamlit` and run:

```bash
pip install streamlit
neuros dashboard

## Constellation Demo Pipeline

In addition to single‑device pipelines, neurOS includes a **Constellation
demo** that ingests and synchronises multiple modalities (EEG, audio,
video, EDA, fNIRS/HD‑DOT, respiration, ECG and phone sensors), writes
raw data to NWB/Zarr (or fallbacks), exports curated samples into
WebDataset shards and optionally launches a distributed training job.
The demo exposes Prometheus metrics for observability and supports
fault injection to test robustness.

To run the demo locally for 10 seconds and store data in
`/tmp/constellation_demo`:

```bash
neuros constellation \
  --duration 10 \
  --output-dir /tmp/constellation_demo \
  --subject-id demo \
  --session-id session1 \
  --fault-injection
```

If you have a Kafka broker running on `localhost:9092`, events will be
published to topics prefixed with `raw`.  Use `--no-kafka` to disable
streaming and run in dry‑run mode.  A helper script
`scripts/run_local_demo.py` simplifies launching the demo and
optionally starting the local Kafka stack via Docker Compose.

See `docs/runbook_constellation.md` for a detailed guide, including
instructions for starting Kafka with Docker Compose, importing the
preconfigured Grafana dashboard and running integration tests.
```

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** – Get up and running in 5 minutes
- **[CONTRIBUTING.md](CONTRIBUTING.md)** – Developer guide and contribution guidelines
- **[AUDIT.md](AUDIT.md)** – Current project status and development roadmap
- **[docs/](docs/)** – Technical documentation and white papers

## Project Status

**Version:** 2.0.0 (Beta)
**Status:** Core functionality complete, production readiness in progress

- ✅ Core pipeline working (single and multi-modal)
- ✅ 10+ models implemented (SimpleClassifier, RandomForest, EEGNet, Transformer, etc.)
- ✅ 15+ drivers for different modalities (EEG, video, motion, EMG, ECG, etc.)
- ✅ CLI and API functional
- ✅ Auto-configuration system
- ⚠️ Test coverage: 40% (target: >90%)
- ⚠️ Model persistence in progress
- ⚠️ Hardware testing needed

See [AUDIT.md](AUDIT.md) for detailed status and roadmap.

## Contributing

Contributions are welcome!  Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines on adding new drivers, processors or models.  The architecture is
designed to be extensible, so new functionality can be added with minimal
boilerplate.

## Support

- **Issues:** Report bugs or request features on [GitHub Issues](https://github.com/shulyalk/neuros-v1/issues)
- **Discussions:** Ask questions and share ideas
- **Documentation:** Check the [docs/](docs/) folder for technical details

## License

This project is licensed under the MIT license – see [LICENSE](LICENSE) for details.

---

**NeurOS v1** – Building the future of brain-computer interfaces 🧠✨