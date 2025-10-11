# Constellation Pipeline Runbook

This runbook describes how to operate the neurOS Constellation pipeline for
ingesting, synchronising, storing, exporting and training on multi‑modal
brain state data.  The pipeline is implemented in the
`neuros.cloud.pipeline_cloud` module and can be invoked via the command‐line
interface.

## Prerequisites

* Python 3.10 or newer
* Required Python packages: see the `setup.py` extras `constellation`
  for a list of dependencies (e.g. `confluent_kafka`, `pynwb`,
  `ome_zarr`, `prometheus_client`, `petastorm`, `webdataset`, etc.).
  Install them with pip, e.g.:

  ```bash
  pip install -e .[constellation]
  ```

* Kafka broker accessible at the host/port specified via
  `--kafka-bootstrap` (default `localhost:9092`).

  For local testing you can spin up a Kafka broker and Zookeeper
  using the provided `docker-compose.yml` in the project root.  To
  start the services run:

  ```bash
  # start Zookeeper and Kafka in the background
  docker-compose up -d
  # verify that Kafka is listening on port 9092
  docker-compose ps
  ```

  To stop the services run `docker-compose down`.  The Compose file
  uses the `confluentinc/cp-zookeeper` and `confluentinc/cp-kafka`
  images and exposes ports `2181` and `9092` on localhost.  If you
  prefer not to use Kafka during testing, pass the `--no-kafka`
  flag to the CLI and the pipeline will operate without sending
  events to Kafka.

* Optional: an AWS account and IAM role configured for SageMaker if you
  intend to launch distributed training jobs.

## Running the demo pipeline

Use the `neuros` CLI with the `constellation` subcommand to execute
the end‑to‑end pipeline.  The command takes parameters for the
ingestion duration, output directory, subject and session IDs,
Kafka configuration and optional fault injection.  For example, to
run a 10 second ingest of all modalities, write the raw data to
`/tmp/constellation_demo`, shard features into WebDataset archives
and launch a (no‑op) SageMaker job:

```bash
neuros constellation \
  --duration 10 \
  --output-dir /tmp/constellation_demo \
  --subject-id subj123 \
  --session-id sessA \
  --kafka-bootstrap localhost:9092 \
  --topic-prefix raw \
  --fault-injection \
  --sagemaker-config path/to/job_config.yaml
```

If you prefer to test the pipeline without a running Kafka broker (for
example when working offline), include the `--no-kafka` flag.  In
this mode the ingestion loop runs as usual but no events are
published; instead a ``NoopWriter`` counts the number of samples.

```bash
neuros constellation \
  --duration 5 \
  --output-dir /tmp/constellation_demo_no_kafka \
  --subject-id demo \
  --session-id test \
  --no-kafka
``` 
### Command‑line options

| Option | Description |
|-------|-------------|
| `--duration` | Duration of ingestion per modality in seconds (default 10) |
| `--output-dir` | Base directory for raw and processed data (simulates S3) |
| `--subject-id` | Identifier for the subject; used in metadata and file names |
| `--session-id` | Identifier for the recording session |
| `--kafka-bootstrap` | Kafka bootstrap servers (host:port) |
| `--topic-prefix` | Prefix for Kafka topics, e.g. `raw` or `aligned` |
| `--fault-injection` | Enable synthetic packet loss and jitter for testing |
| `--sagemaker-config` | Path to a JSON/YAML configuration file for the training job |

### Pipeline stages

1. **Ingestion**: Simulated drivers for EEG, audio, video, EDA,
   fNIRS/HD‑DOT, respiration, ECG and phone behavioural metrics are
   instantiated.  Each driver emits timestamped samples at its own
   sampling rate.  A `KafkaWriter` publishes each event to
   topics named `raw.<modality>.<sensor_id>`.  Prometheus metrics
   capture throughput and latency.  Fault injection can drop packets
   or perturb timestamps to exercise the sync layer.

2. **Raw storage**: After ingestion completes, the collected samples
   are written to cloud‑native formats:
   * Time series modalities (EEG, EDA, respiration, ECG, fNIRS, phone,
     audio) are stored in NWB files with per‑channel data arrays.
   * Video frames are stacked and saved as an OME‑Zarr hierarchy.

3. **Feature export**: The directory of raw files is sharded into
   WebDataset tar archives using the `export_to_webdataset` helper.
   Optionally a Petastorm dataset can be generated (commented out in
   the pipeline).  The resulting shards reside under
   `<output-dir>/gold/<subject>/<session>/`.

4. **Training**: The `launch_training` wrapper submits a SageMaker
   distributed data‑parallel (DDP) job configured via the provided
   YAML/JSON file.  When run outside AWS or without the `sagemaker`
   package installed, the function logs the configuration and returns.

## Observability

The ingestion loop exposes Prometheus counters and histograms for
total samples ingested, per‑modality latency and Kafka stream
latency.  Point your Prometheus server at the default metrics
endpoint to scrape these metrics and visualise them in Grafana.

## Fault injection

Pass the `--fault-injection` flag to simulate ~5 % packet loss and
timestamp jitter.  Use this mode during dry runs to test
synchronisation and error handling.

## Dry runs and testing

To perform a dry run in an instrumented space, follow these steps:

1. Ensure all required sensors are connected and streaming.  For this
   demo the drivers emit synthetic data, but in a real deployment the
   drivers should wrap hardware SDKs or LabStreamingLayer (LSL) outlets.

2. Start a local Kafka broker and (optionally) a Prometheus server.

3. Run the demo pipeline with a short duration (e.g. `--duration 5`) and
   verify that NWB and Zarr files are created under the output
   directory.  Use the provided CLI to run additional tests or
   integrate with existing notebooks.

4. Examine the metrics via `curl http://localhost:8000/metrics` if the
   application is instrumented accordingly and observe latency and
   throughput statistics.

5. Use the exported WebDataset shards as input to your model
   training pipeline.  The example `sagemaker_launcher.py` shows how
   to configure a distributed training job.

## Extending for new modalities

To integrate additional devices (e.g. HD‑DOT, eye tracking, phones
sensors), implement a new driver in `neuros/drivers` inheriting from
`BaseDriver`, then update the `drivers` dictionary in
`pipeline_cloud.run_multimodal_ingestion` to include your driver.  The
rest of the pipeline will handle ingestion, streaming, storage and
export automatically.

---

For more information, see the module documentation in
`neuros/cloud/pipeline_cloud.py` and the docstrings of individual
drivers and utility functions.

## Additional utilities

The repository includes a few helper scripts and dashboards to make
development and testing easier:

* **`scripts/run_local_demo.py`** – A wrapper that optionally starts
  the local Kafka stack using Docker Compose and then runs the
  Constellation demo with sensible defaults.  Pass `--with-kafka`
  to bring up Kafka via Compose or `--no-kafka` to skip Kafka.

* **`scripts/test_pipeline.py`** – A simple integration test that
  runs a short dry‑run ingestion and verifies that raw NWB/Zarr files
  and WebDataset shards are produced.  Use this script to sanity
  check your installation.

* **`monitoring/grafana_dashboard.json`** – A preconfigured Grafana
  dashboard that visualises the Prometheus metrics emitted by the
  ingestion loop.  Import this JSON file into Grafana to monitor
  throughput and latency.