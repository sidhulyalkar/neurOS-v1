"""Build a deployment-unit-disjoint evidence manifest without downloading data.

The arrays are synthetic so this example stays deterministic and CI-friendly.
The metadata shape mirrors the conventional MOABB (X, labels, metadata) boundary;
replace the fixture with a real MOABB paradigm result for an actual study.
"""

from __future__ import annotations

import json

import numpy as np

from neuros.foundation_models import (
    GroupedEvaluationData,
    find_evidence_sources,
    hold_out_groups,
)

rng = np.random.default_rng(11)
records = []
labels = []
trials = []
for subject in ("1", "2", "3"):
    for session in ("0", "1", "2"):
        for trial in range(6):
            label = trial % 2
            signal = rng.normal(size=(8, 64))
            signal[0] += 0.25 * label
            trials.append(signal)
            labels.append(label)
            records.append(
                {
                    "subject": subject,
                    "session": session,
                    "run": "0",
                }
            )

bundle = GroupedEvaluationData.from_moabb_result(
    (np.asarray(trials), np.asarray(labels), records),
    dataset_id="example-longitudinal-eeg",
)
partition = hold_out_groups(
    bundle,
    split_unit="session",
    held_out_values=["2"],
)
protocol = partition.protocol(
    name="example-held-out-session",
    transfer_regime="linear_probe",
    notes=("synthetic example only; replace with a pinned public dataset",),
)

print("Curated longitudinal EEG sources")
for source in find_evidence_sources(role="longitudinal_bci"):
    print(f"  {source.id}: {source.title}")

print("\nPre-model evidence manifest")
print(json.dumps(partition.manifest(protocol=protocol), indent=2, sort_keys=True))
