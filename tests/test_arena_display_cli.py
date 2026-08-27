from __future__ import annotations

import sys

import pytest

from neuros.arena.display_cli import main


def test_measured_display_cli_requires_explicit_units_before_file_access(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "neuros-arena-display",
            "--manifest",
            "does-not-exist.json",
            "--observation",
            "does-not-exist.csv",
            "--epoch",
            "0",
            "--evidence-class",
            "measured_photodiode",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
