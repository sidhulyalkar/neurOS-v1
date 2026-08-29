"""Project scaffolding for the supported neurOS developer on-ramp."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neuros.errors import ConfigurationError


SUPPORTED_PROJECT_TEMPLATES = ("mock-bci",)

_MOCK_BCI_CONFIG = """schema_version: 1

metadata:
  name: neurOS-starter
  purpose: deterministic local runtime and qualification starter

streams:
  - id: eeg
    source:
      plugin: mock
      options:
        sampling_rate: 250.0
        channels: 8
    transforms:
      - plugin: smoothing
        options:
          window_size: 3

decoder:
  plugin: threshold
  options:
    threshold: 0.0

runtime:
  queue_capacity: 16
  overflow_policy: drop_oldest

sinks: []
monitors: []
"""

_PROJECT_README = """# neurOS starter project

This project is deliberately small. It exercises the maintained neurOS runtime,
record/replay path, and software-qualification boundary without downloading a
dataset, training a model, or implying a biological or hardware claim.

## 1. Inspect the environment

```bash
neuros doctor
neuros compatibility
```

## 2. Validate and run

```bash
neuros validate neuros.yaml
neuros run neuros.yaml --duration 2
```

## 3. Produce a reproducible software-evidence bundle

```bash
neuros qualify neuros.yaml --output evidence/qualification --duration 1
neuros reproduce evidence/qualification
```

The qualification bundle is software evidence. It does not establish neural
model efficacy, hardware validity, closed-loop performance, safety, or clinical
benefit.

## Next steps

- Replace the mock source through a maintained BrainFlow, LSL, replay, or dataset
  integration rather than editing the neurOS kernel.
- Add an external decoder through the plugin/qualification interfaces.
- Use NSQ for leakage-controlled comparative neural-system evidence.
- Use ORION only when you need governed neural representations, tokenization, or
  adaptation on top of the same evidence authority.

Project documentation: https://github.com/sidhulyalkar/neurOS-v1
"""

_PROJECT_GITIGNORE = """.venv/
__pycache__/
*.py[cod]
.pytest_cache/
evidence/
sessions/
artifacts/
"""


def _template_files(template: str) -> dict[str, str]:
    if template != "mock-bci":
        raise ConfigurationError(
            f"Unsupported project template {template!r}; "
            f"choose one of {', '.join(SUPPORTED_PROJECT_TEMPLATES)}"
        )
    return {
        "neuros.yaml": _MOCK_BCI_CONFIG,
        "README.md": _PROJECT_README,
        ".gitignore": _PROJECT_GITIGNORE,
    }


def init_project(
    destination: str | Path,
    *,
    template: str = "mock-bci",
    force: bool = False,
) -> dict[str, Any]:
    """Create a minimal, runnable neurOS project without deleting user files.

    ``force`` permits replacement of neurOS-managed starter files only. Unrelated
    files in an existing directory are preserved.
    """

    root = Path(destination).expanduser().resolve()
    files = _template_files(template)

    if root.exists() and not root.is_dir():
        raise ConfigurationError(f"Project destination is not a directory: {root}")

    existing_managed = [name for name in files if (root / name).exists()]
    if existing_managed and not force:
        joined = ", ".join(sorted(existing_managed))
        raise ConfigurationError(
            f"Project destination already contains neurOS starter files: {joined}. "
            "Use --force to replace only those managed files."
        )

    root.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    replaced: list[str] = []
    for relative, content in files.items():
        path = root / relative
        existed = path.exists()
        path.write_text(content, encoding="utf-8")
        (replaced if existed else created).append(relative)

    return {
        "schema_version": 1,
        "project_root": str(root),
        "template": template,
        "config": str(root / "neuros.yaml"),
        "created": sorted(created),
        "replaced": sorted(replaced),
        "next_commands": [
            "neuros doctor",
            "neuros validate neuros.yaml",
            "neuros run neuros.yaml --duration 2",
            "neuros qualify neuros.yaml --output evidence/qualification --duration 1",
            "neuros reproduce evidence/qualification",
        ],
        "evidence_boundary": (
            "starter workflow produces software/runtime evidence only; it does not "
            "qualify neural efficacy, hardware, closed-loop behavior, safety, or clinical benefit"
        ),
    }


__all__ = ["SUPPORTED_PROJECT_TEMPLATES", "init_project"]
