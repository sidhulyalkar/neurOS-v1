# Braindecode interoperability

neurOS integrates Braindecode as an **optional decoder ecosystem**, not as code to copy into the neurOS model package.

The division of responsibility is deliberate:

| Layer | Authority |
| --- | --- |
| acquisition, clocks, replay, queues | neurOS |
| `SignalFrame` → `NeuralWindow` provenance | neurOS |
| published architecture implementation | Braindecode |
| Skorch-based training loop | Braindecode `EEGClassifier` |
| protocol identity and split authority | neurOS Evidence |
| subject/session/device transfer claims | neurOS Evidence |
| activation-path/intervention claims | only after neurOS mechanistic qualification |

## Install

Braindecode 1.7 requires Python 3.11 or newer. The neurOS kernel remains Python 3.10 compatible because the integration is optional and lazily imported.

```bash
pip install "neuros[braindecode]"
```

or for the model package alone:

```bash
pip install "neuros-models[braindecode]"
```

The currently qualified upstream release line is pinned to `braindecode>=1.7,<1.8` so a future upstream API change cannot silently widen the neurOS compatibility claim.

## Initial model surface

`BraindecodeDecoder` initially accepts only:

- `EEGNet`
- `EEGConformer`
- `ShallowFBCSPNet`
- `Deep4Net`

This is intentionally a whitelist. Braindecode contains many additional architectures, including pretrained/foundation models with architecture-specific preprocessing, sampling-rate, channel-location, or distribution requirements. Their presence upstream does not make them automatically supported by neurOS.

```python
from neuros.models import BraindecodeDecoder

model = BraindecodeDecoder(
    "EEGNet",
    n_channels=8,
    n_times=500,
    n_classes=4,
    sample_rate_hz=250.0,
    n_epochs=20,
    batch_size=32,
)
```

The adapter requires exact input geometry:

```text
(batch, channels, time)
```

It never silently:

- resamples;
- crops or pads windows;
- changes channel ordering;
- bandpass-filters;
- scales units;
- constructs an MNE montage or sensor geometry;
- chooses calibration/evaluation splits.

Those transformations must be explicit upstream of the decoder and therefore reproducible in replay/evidence artifacts.

## Runtime path

The intended online/offline execution path is the same native RuntimeGraph used by neurOS models:

```text
MNE / LSL / BrainFlow / replay
              |
              v
         SignalFrame
              |
              v
    window transform
              |
              v
         NeuralWindow
      (channel, time)
              |
      RuntimeExecutor
      adds batch axis
              |
              v
   BraindecodeDecoder
              |
 upstream EEGClassifier
              |
              v
        DecoderOutput
      + window provenance
```

A decoder prediction therefore remains traceable to the exact window ID, time interval, channel names, sample rate, quality state, and source `SignalFrame.sequence_id` values that produced it.

## Training authority

neurOS does not maintain a second imitation trainer. `BraindecodeDecoder.train()` constructs the selected upstream model and delegates optimization to `braindecode.EEGClassifier`.

The adapter makes the following training choices explicit by default:

- `torch.nn.CrossEntropyLoss`;
- `torch.optim.AdamW`;
- explicit learning rate and weight decay;
- explicit epoch count and batch size;
- `train_split=None` inside the adapter.

`train_split=None` is intentional. neurOS Evidence, not an opaque model wrapper, should own subject/session/calibration/validation partition authority. A research protocol that needs early stopping or a validation set should build that split explicitly rather than letting an adapter derive one from evaluation data.

## What the integration evidence means

The compatibility lane establishes an **integration** claim when it is green:

1. the real pinned Braindecode distribution installs on supported Python versions;
2. every whitelisted architecture can consume canonical neurOS window geometry;
3. EEGNet can train through the upstream `EEGClassifier` path;
4. a trained upstream EEGNet can execute through `SignalFrame → NeuralWindow → RuntimeGraph`;
5. probabilities and upstream version identity reach `DecoderOutput`;
6. Python 3.10 remains free of the optional dependency.

That does **not** establish model utility on a neuroscience task.

## Paired longitudinal evidence

neurOS now has a dedicated paired evidence contract for comparing maintained native EEGNet with upstream Braindecode EEGNet without giving either implementation authority over the evaluation split.

`run_external_task_decoder_case(...)` restores the same serialized `LongitudinalCaseAuthority` used by the native decoder lane. `pair_task_performance(...)` refuses to create a paired result unless both runs agree on the authority fingerprint, processed-data SHA-256, partition fingerprint, calibration-split fingerprint, class vocabulary, sample counts, and calibration-budget set.

The pair records:

- native and upstream learned-state SHA-256 values;
- adapter/method fingerprints;
- accuracy and balanced accuracy;
- ROC-AUC where defined;
- Brier score and expected calibration error;
- fit and inference cost;
- explicit sampling frequency;
- whether representation or mechanistic evidence is actually available.

Braindecode representation and mechanistic fields remain unavailable rather than being fabricated to match the richer native decoder schema.

The real-data runner is:

```bash
python scripts/evidence/run_moabb_braindecode_pair.py \
  --dataset kumar2024 \
  --subjects 1 \
  --model-seeds 101,503,1601 \
  --budgets 0,1,2,5,10 \
  --history-policy prior \
  --fmin 8 \
  --fmax 30 \
  --resample 128 \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --device cpu \
  --output evidence-run/study
```

It emits `study_manifest.json`, `split_authority.json`, native/external/paired run JSON, paired CSV results, a descriptive summary/report, and SHA-256 declarations for the complete bundle. Optimization seeds are collapsed within subject/session case before case-level summaries so repeated training seeds are not counted as independent deployment units.

## Running the real-data study in GitHub Actions

The manual workflow **neurOS Braindecode paired real-data study** exposes the study authority as explicit `workflow_dispatch` inputs. It supports the maintained longitudinal MOABB dataset keys `kumar2024`, `ma2020`, `lee2019-mi`, and `wang2026`.

The workflow intentionally does not run a real dataset on every pull request. Pull requests execute only a dependency-light contract check that proves the runner remains discoverable without installing Braindecode. A manual study run installs the pinned Braindecode/MOABB stack, records Python/package/Git identity, downloads data through the existing MOABB path, executes the paired study, verifies every declared artifact hash and authority fingerprint, then uploads the complete bundle for 90 days.

A green manual workflow is **execution evidence**, not an automatic scientific promotion. Before a result is summarized publicly, inspect the uploaded `study_manifest.json`, `split_authority.json`, `summary.json`, `report.md`, and per-run learned-state/configuration fingerprints. Multi-subject claims should use participant-aware repeated-measures inference rather than treating subject/session cases or optimization seeds as independent participants.

## Mechanistic interpretation boundary

The adapter exposes the trained upstream PyTorch module via `analysis_model()`, but its `InterpretabilityManifest` intentionally declares no stable neurOS hook paths yet.

That distinction prevents a common failure mode: a PyTorch module being technically hookable is not the same as an architecture having stable, semantically meaningful, version-qualified intervention surfaces.

A Braindecode architecture should become `mechint_ready` only after neurOS records and tests named surfaces for the pinned upstream version, then validates capture/replacement behavior and causal intervention semantics.

## Foundation-model caution

Pretrained EEG models can have strict training-distribution assumptions. For example, an upstream checkpoint may assume a particular sampling rate, physical unit scale, channel representation, or filtering regime. neurOS should encode these as explicit adapter/evidence requirements rather than silently normalizing arbitrary input until the model accepts it.

The correct promotion sequence is:

```text
upstream model exists
        |
        v
shape/API integration
        |
        v
preprocessing distribution contract
        |
        v
real-dataset transfer evidence
        |
        v
mechanistic evidence
        |
        v
hardware / closed-loop evidence
```

A model can move upward only when the corresponding executable evidence exists.
