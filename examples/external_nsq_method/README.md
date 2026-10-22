# External NSQ method example

This is a deliberately small, non-promoted example of an external method
participating in the neurOS Neural System Qualification (NSQ) referee.

It is ordinary scikit-learn code:

```text
StandardScaler -> LogisticRegression
```

The fixture is deterministic and synthetic. Its scores are software-contract
evidence only; this example makes no claim about neural decoding efficacy,
biological meaning, competitiveness, or the frozen Kumar2024 comparison.

## Run it

From the repository root, after installing the repository's research/test
dependencies (including `scikit-learn`):

```bash
python -m examples.external_nsq_method.demo
pytest -q examples/external_nsq_method/test_example.py
pytest -q examples/external_nsq_method/test_external_failures.py
```

`test_external_failures.py` is a black-box conformance check: deliberately
faulty external decoders (non-finite or wrongly shaped `predict_proba`, wrong
probability class order, a raising `fit`) must yield failure-preserving
per-budget rows with `failure_type`/`failure_reason` and no score, while the
final-assessment partition identity is unchanged. A dataset-lineage mismatch
must abort at preflight before any decoder is created.

The example tests are intentionally **not** part of the default `pytest.ini`
discovery (`testpaths = tests`), so the optional scikit-learn dependency never
affects unrelated suites. Run them explicitly with the command above.

## Use it from another repository

`demo.py` uses a relative import when run as part of the `examples` package
(`__package__` is set) and a plain sibling import when run as a script.

First install neurOS following the repository's
[installation instructions](../../README.md#get-useful-evidence-in-minutes)
(until public packages are published, this means installing from a clone):

```bash
git clone https://github.com/sidhulyalkar/neurOS-v1.git
cd neurOS-v1
python -m venv .venv
source .venv/bin/activate
python scripts/bootstrap.py --profile bci --test-tools
python -m pip install -e packages/neuros-foundation
pip install scikit-learn
```

Windows PowerShell equivalent:

```powershell
git clone https://github.com/sidhulyalkar/neurOS-v1.git
Set-Location neurOS-v1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python scripts/bootstrap.py --profile bci --test-tools
python -m pip install -e packages/neuros-foundation
python -m pip install scikit-learn
```

If `Activate.ps1` is blocked by the execution policy, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that shell first.

Then, with that environment active, copy the two files into your own project:

```bash
mkdir ../my-nsq-method && cd ../my-nsq-method
cp ../neurOS-v1/examples/external_nsq_method/method.py .
cp ../neurOS-v1/examples/external_nsq_method/demo.py .
python demo.py
```

Windows PowerShell equivalent:

```powershell
New-Item -ItemType Directory ..\my-nsq-method; Set-Location ..\my-nsq-method
Copy-Item ..\neurOS-v1\examples\external_nsq_method\method.py .
Copy-Item ..\neurOS-v1\examples\external_nsq_method\demo.py .
python demo.py
```

Note: in Windows PowerShell 5.1, `python demo.py > out.json` writes UTF-16
(and `Out-File -Encoding utf8` adds a BOM). To capture BOM-free UTF-8 JSON,
write the file directly from Python:

```powershell
python -c "import json, demo; open('out.json', 'w', encoding='utf-8').write(json.dumps(demo.run_demo(), indent=2, sort_keys=True))"
```

or use .NET, whose `WriteAllText` default is UTF-8 without BOM:

```powershell
[IO.File]::WriteAllText("$PWD\out.json", ((python demo.py) -join "`n"))
```

To also run the failure-conformance tests standalone, copy
`test_external_failures.py` next to the other two files and run
`python -m pytest -q test_external_failures.py`.

Keep the public `neuros.foundation_models` imports and replace
`ExternalDecoder` with your own estimator implementation.

## Provenance note: placeholder hashes

The dataset lineage, preprocessing-authority, and calibration-authority
hashes in `demo.py` are **placeholders**: SHA-256 digests of fixed label
strings (e.g. `"synthetic-external-nsq-method-v1:dataset-lineage"`). They
exist only so the referee's identity checks can run on the synthetic fixture.
They are **not** verified dataset or preprocessing hashes and attest to no
real data. A real qualification must supply lineage hashes computed from the
actual source data and preprocessing authorities.

## Contract boundary

The external implementation owns:

- feature transforms and estimator settings;
- fitting and prediction;
- probability semantics and class-column order;
- deterministic learned-state serialization and hashing.

neurOS owns:

- the frozen protocol and dataset lineage;
- source, calibration, and untouched final-assessment observation roles;
- the calibration budget frontier;
- score semantics, run identity, provenance, and failure preservation.

The factory returns a fresh estimator for every qualification budget. The
content-addressable learned state is an explicit SHA-256 over the fitted
scikit-learn arrays and class labels; no pickle serialization is used.

The method specification records the installed scikit-learn version. This is
intentional provenance: changing that dependency changes the method-spec SHA,
even when the source code and estimator settings are unchanged.
