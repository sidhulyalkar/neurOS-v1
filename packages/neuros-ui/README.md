# neurOS UI

`neuros-ui` contains optional dashboard, API, and visualization integrations for neurOS.

> **Maturity boundary:** this package currently builds as part of the workspace, but it does not yet have the same release-blocking behavioral qualification as `neuros-core`. Treat it as an integration/prototyping surface until dedicated dashboard/API contract tests and supported reference deployments are added.

## Installation

```bash
pip install neuros-ui
pip install "neuros-ui[dashboard]"  # Streamlit
pip install "neuros-ui[api]"        # FastAPI + Uvicorn
pip install "neuros-ui[viz]"        # Plotly + Seaborn
pip install "neuros-ui[all]"
```

## Architectural role

UI code should consume the same neurOS configuration, runtime, recording, and event contracts used by local execution. It should **not** become a second orchestration engine with different lifecycle, timing, or model semantics.

The intended direction is:

```text
RuntimeGraph / recording / quality / model evidence
                    |
                    v
              API / dashboard
```

High-value UI work for the Developer Preview includes:

- runtime graph and stream health;
- queue loss and latency distributions;
- recording/replay inspection;
- device/model/config provenance;
- qualification/evidence history;
- model/representation comparison artifacts;
- explicit warning surfaces when a claim has only software, synthetic, or dataset-level evidence.

## Product rule

Real-time BCI execution should remain local by default. A dashboard may observe, configure, or request actions through stable APIs, but UI availability must never become a hidden prerequisite for acquisition or safety-critical runtime behavior.

## Documentation

Current platform documentation and package maturity live in:

- `docs/PROJECT_STATUS.md`
- `docs/ARCHITECTURE.md`
- `docs/API_REFERENCE.md`
- `ROADMAP.md`

Repository: https://github.com/sidhulyalkar/neurOS-v1

## License

MIT License.
