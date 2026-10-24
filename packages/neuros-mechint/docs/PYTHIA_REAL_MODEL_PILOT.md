# First real-checkpoint Circuit Auditor pilot

**Status:** runnable experimental protocol, not an executed real-model result.
**Owner issue:** [#201](https://github.com/sidhulyalkar/neurOS-v1/issues/201).
**Dependency:** PRs #199 and #200, now merged on main.

This is a measurable first real-weight study, **not** a validation of LLM
interpretability in general and not evidence of a biological circuit.

## Frozen v1 design

| Field | Preregistered value |
| --- | --- |
| Model | EleutherAI/pythia-70m-deduped (Apache-2.0) |
| Model/tokenizer revision | Same **immutable 40-character HF commit**, required at runtime |
| Scoring | Correct versus held-out foil next-token logit margin |
| Prompt pattern | \`A B C D A B C\` → \`D\` (versus \`E\`) |
| Split | 4 discovery, 4 validation; disjoint lexical token pools |
| Component universe | Six attention outputs and six MLP outputs of GPT-NeoX |
| Discovery | Rank individual zero-ablation effects on discovery *only* |
| Candidates | Top two effects versus bottom-two decoy |
| Validation | Frozen candidates, held-out examples |
| Controls | 24 random equal-size circuits per example; activation-magnitude comparator |
| Interventions | Zero and global scalar discovery-mean replacement |
| Synthetic task gate | At least 75% positive next-token margin in discovery, then validation |
| Evidence tier | \`SCIENTIFIC_SYNTHETIC\` (real checkpoint but constructed stimuli) |
| Output | Frozen dataset input manifest, positive/decoy evidence artifacts, status record |

The null expectation is permitted: the model may fail even the initial task.
When the discovery task gate fails, the runner writes \`dataset_manifest.json\`
and \`status.json\` with \`BASELINE_NOT_QUALIFIED\`; it does **not** run or
promote circuit tests. Validation model *outputs* are not inspected until the
discovery candidate is frozen and evaluated on held-out trials.

Even if the model passes, a causal intervention on a full attention/MLP
module is a **coarse model-component intervention**, not a discovered
attention-head circuit. The methodology currently has no gradient EAP-IG
comparator, conditional donor, or human-neural counterpart.

## Manually dispatched public-checkpoint qualification

The [Pythia empirical pilot GitHub Actions workflow](../../../.github/workflows/mechint-pythia-empirical-pilot.yml)
is intentionally **manual-only**. No PR check or ordinary push will download
external model weights or consume a large inference budget.

In GitHub, open **Actions → Pythia Circuit Auditor empirical pilot → Run
workflow**, select `main` and leave `stage=preflight`. It uses the
verified immutable public checkpoint commit
`e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` by default, subject to
a live Hugging Face exact-revision check. The CPU environment pins PyTorch
2.4.1, Transformers 4.45.2 and huggingface-hub 0.25.2. If the public
checkpoint has become inaccessible or incompatible, the workflow must fail
closed instead of substituting another checkpoint or changing the selectors.

The preflight only checks the configured six-layer hook surface and four
discovery examples. It does **not** expose validation model outputs, select a
causal circuit or estimate intervention effects. It emits
`PREFLIGHT_ONLY_QUALIFIED` or `BASELINE_NOT_QUALIFIED` plus the input
manifest. A baseline failure is retained as a negative experimental result.

Only if that checkpoint passes the discovery behavioral gate should a second
manual run select `stage=full`. Full execution runs the **unchanged v1**
study, saves both candidate/decoy evidence artifacts or negative v2 artifacts,
and checks their content hashes before creating a versioned GitHub Actions
artifact bundle. On failure, environment manifests, logs and any complete
diagnostic files are still uploaded for investigation.

**No empirical results are claimed by enabling the workflow.** An Actions
"success" means the protocol completed and artifacts were verified. It does
not imply promotion, mechanism discovery, model task validity on held-out
examples or independent scientific reproduction. Keep the exact source commit,
checkpoint, Python/package versions, seed and evidence hashes together when
requesting reproduction from another environment.

## Run locally

Python 3.10–3.12, CPU (slower) or an appropriate NVIDIA GPU:

\`\`\`bash
python scripts/bootstrap.py --profile kernel --test-tools
pip install -e "packages/neuros-mechint[dev,real-model-pilot]"
\`\`\`

Resolve an immutable revision **once and record it before running**:

\`\`\`bash
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('EleutherAI/pythia-70m-deduped').sha)"
\`\`\`

Copy the full 40-character SHA into the invocation below. Do not use
\`main\`, \`step143000\`, or an unpinned model alias as a revision.

\`\`\`bash
python packages/neuros-mechint/examples/24_pythia_real_model_pilot.py \
  --revision <40-character-model-commit> \
  --device cpu \
  --out ./pythia-audit-E001
\`\`\`

For NVIDIA use \`--device cuda\`; \`--device mps\` is available on supported
Mac hardware. If the selected Transformers version changes the structured
GPT-NeoX attention output, the preflight fails rather than silently selecting
a different tensor leaf. Pin the package versions from the execution report
for independent reproduction.

Verify the exported artifacts:

\`\`\`bash
neuros-mechint verify-evidence-artifact ./pythia-audit-E001/high-effect-discovery.json --json
neuros-mechint verify-evidence-artifact ./pythia-audit-E001/low-effect-decoy.json --json
\`\`\`

The commands above are intended for runs that pass the discovery preflight.
The runner has **not** been executed here with downloaded pretrained weights,
so no results are asserted. A fully nonestimable intervention protocol is
recorded as `NONESTIMABLE` in `status.json` **and** preserved as a
hash-verifiable evidence-pack artifact: invalid cases keep their original
failure reasons and unmeasured effect statistics are serialized as JSON
`null`, never as numeric zero. This is a scientifically meaningful negative
diagnostic, not an automatically recovered or successful analysis.

In the existing evidence-pack schema, `publication_ready` means that
reproducibility revisions are pinned. It **does not** imply that the candidate
passed causal faithfulness, that the validation task succeeded, or that
independent reproduction occurred. The pilot's top-level status distinguishes
these concerns explicitly.

## Scientific limitations and closure

1. **Pretraining contamination:** held-out means unseen by the *mechanism discovery
   procedure*, not unseen during pretraining. The model may know the words.
2. **Baseline confounding:** an expected-token margin above a foil is a weak
   induction proxy. It does not establish that the model copied a token by
   induction. Report this limitation regardless of score.
3. **Distribution shift:** zero and scalar global-mean activation replacements
   are not guaranteed on-manifold; mean replacement is not a matched empirical
   donor. Sensitivity must be reported, not discarded.
4. **Null distribution:** 24 random-control draws yield a minimum one-sided
   plus-one p-value of 1/25, *before* any multiplicity adjustment. The tool's
   percentile is not by itself a calibrated hypothesis test.
5. **Selection bias:** two candidates (high-effect and decoy) are constructed.
   Their comparison is descriptive without multiplicity control.
6. **Scope of truth:** a held-out passed promotion policy is mechanistic
   *evidence under explicit interventions*, not proof of a faithful circuit,
   biological homology, or independent reproduction.
7. **Closure requirement:** another environment must reproduce the qualitative
   decision with preregistered numeric tolerances, using the same fixed model,
   tokenizer, prompts, method versions and semantics. The next iteration should
   compare EAP-IG and a matched non-zero donor, then repeat on neural data.

The [Pythia model card](https://huggingface.co/EleutherAI/pythia-70m-deduped)
describes its Apache-2.0 license, checkpoint family, intended research use
and limitations. Do not use this pilot for human-facing model deployment.
