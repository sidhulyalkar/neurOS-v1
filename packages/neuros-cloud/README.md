# neurOS Cloud

`neuros-cloud` contains optional distributed-data, storage, observability, and cloud-provider integrations for neurOS.

> **Maturity boundary:** this package currently builds as part of the workspace, but it is not the neurOS real-time execution kernel and does not yet have provider-specific release qualification. Treat cloud integrations as optional infrastructure until their individual deployment paths have dedicated contract and reference-environment tests.

## Installation

```bash
pip install neuros-cloud
pip install "neuros-cloud[kafka]"      # Kafka / ZeroMQ / Redis integrations
pip install "neuros-cloud[aws]"        # AWS + SageMaker dependencies
pip install "neuros-cloud[export]"     # WebDataset / OME-Zarr / Arrow
pip install "neuros-cloud[monitoring]" # Prometheus / MLflow / W&B
pip install "neuros-cloud[all]"
```

## Architectural role

Cloud infrastructure should sit above stable neurOS runtime and artifact contracts:

```text
local acquisition / RuntimeGraph / recording / replay
                         |
                         v
              evidence + artifact boundary
                         |
                         v
       registry / telemetry / orchestration / collaboration
```

Real-time neural acquisition, queue semantics, timing, and safety behavior should remain local by default. A cloud outage must not silently alter the meaning of a running neural pipeline.

High-value cloud/product work includes:

- immutable experiment/model/data/evidence artifact registry;
- remote experiment configuration and launch requests over stable APIs;
- qualification history for named hardware/model/config combinations;
- fleet/device compatibility metadata;
- aggregate observability and audit trails;
- multi-site collaboration and permissions;
- reproducible benchmark orchestration;
- optional large-scale training jobs that return versioned artifacts to the local runtime.

## Evidence rule

Provider availability is not BCI qualification. Any cloud path that participates in a scientific or operational claim should record provider/runtime versions, configuration, artifact hashes, failure behavior, and the strongest evidence tier actually tested.

## Documentation

Current platform documentation and package maturity live in:

- `docs/PROJECT_STATUS.md`
- `docs/ARCHITECTURE.md`
- `docs/API_REFERENCE.md`
- `ROADMAP.md`

Repository: https://github.com/sidhulyalkar/neurOS-v1

## License

MIT License.
