"""Reviewer-visible defect taxonomy and opaque cases for the neurOS swarm benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._canonical import require_nonempty, require_sha256
from .swarm import SealedSwarmTask

BENCHMARK_SCHEMA = "neuros.scientific_engineering_swarm_benchmark.v1"
# Commitment to the complete scorer corpus, including hidden labels. The public module
# intentionally stores only the digest, never the answer key used to derive it.
BENCHMARK_CORPUS_SHA256 = "122d862b2d43078723b858d5d5520a4ffb24e9e497d0090a52865c20f3aa5e0a"


@dataclass(frozen=True, slots=True)
class DefectDefinition:
    defect_id: str
    category: str
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "defect_id", require_nonempty(self.defect_id, name="defect_id"))
        object.__setattr__(self, "category", require_nonempty(self.category, name="category"))
        object.__setattr__(
            self,
            "description",
            require_nonempty(self.description, name="description"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "defect_id": self.defect_id,
            "category": self.category,
            "description": self.description,
        }


DEFECT_TAXONOMY: tuple[DefectDefinition, ...] = (
    DefectDefinition("D01", "reproducibility", "Evidence belongs to a different Git revision than the candidate being qualified."),
    DefectDefinition("D02", "scientific_validity", "Train and evaluation partitions share participant identity or otherwise violate subject-disjoint evaluation."),
    DefectDefinition("D03", "scientific_validity", "Inference treats lower-level samples as independent while the scientific unit is the participant."),
    DefectDefinition("D04", "reproducibility", "Learned model state is substituted without matching the state identity bound by the authority."),
    DefectDefinition("D05", "reproducibility", "Runtime environment differs from the frozen environment authority."),
    DefectDefinition("D06", "security", "Duplicate JSON keys permit parser-dependent or last-key-wins reinterpretation of a sealed field."),
    DefectDefinition("D07", "security", "Import or file resolution can be redirected to an unbound shadow file instead of the intended implementation."),
    DefectDefinition("D08", "implementation", "CI omits a dependency or plugin required by the repository's own test configuration."),
    DefectDefinition("D09", "reproducibility", "A supposedly complete fleet or evaluation is missing an expected terminal shard or case."),
    DefectDefinition("D10", "reproducibility", "The same logical lease or shard identity is represented more than once."),
    DefectDefinition("D11", "scientific_validity", "Execution changes a preregistered or frozen random seed."),
    DefectDefinition("D12", "scientific_validity", "Provider, retry, or scheduling decisions adapt to scientific outcomes."),
    DefectDefinition("D13", "scientific_validity", "Target-derived information is included in model inputs, features, or preprocessing."),
    DefectDefinition("D14", "scientific_validity", "A learned preprocessing transform is fit using held-out evaluation data."),
    DefectDefinition("D15", "reproducibility", "An artifact lacks enough provenance to bind code, data, model, or configuration identity."),
    DefectDefinition("D16", "reproducibility", "A stochastic operation that affects evaluation is left unseeded or otherwise non-replayable."),
    DefectDefinition("D17", "scientific_validity", "Reported uncertainty or hypothesis testing uses a unit of analysis inconsistent with the claim."),
    DefectDefinition("D18", "reproducibility", "Retry or attempt count exceeds the frozen retry ceiling."),
    DefectDefinition("D19", "reproducibility", "An output is accepted by path/name/existence without a cryptographic binding to its generating assignment."),
    DefectDefinition("D20", "scientific_validity", "The stated scientific claim is stronger than the experiment or evidence can support."),
)
DEFECT_IDS = frozenset(defect.defect_id for defect in DEFECT_TAXONOMY)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    stimulus: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", require_nonempty(self.case_id, name="case_id"))
        object.__setattr__(self, "stimulus", require_nonempty(self.stimulus, name="stimulus"))

    def to_public_dict(self) -> dict[str, str]:
        return {"case_id": self.case_id, "stimulus": self.stimulus}


_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase("case-001", "A PR candidate is at commit 92f0... but the attached passing workflow receipts were generated for commit 1a7c.... The reviewer marks the current candidate qualified without rerunning checks."),
    BenchmarkCase("case-002", "EEG epochs from every participant are pooled, shuffled, and split 80/20 by row. Epochs from participant 7 appear in both train and test partitions."),
    BenchmarkCase("case-003", "A paper-level claim is about generalization across people, but confidence intervals are computed by bootstrapping 48,000 epochs after concatenating all 18 participants. No participant-level estimate is formed."),
    BenchmarkCase("case-004", "An authority records weights_sha256=aaa..., but the worker loads a checkpoint with weights_sha256=bbb... because it has a better validation curve. The artifact is then resealed under the original authority identifier."),
    BenchmarkCase("case-005", "The frozen environment names package version 2.4.1 and CUDA 12.4. A worker actually runs package version 2.5.0 and CUDA 12.6, yet reuses the older environment authority SHA."),
    BenchmarkCase("case-006", "A raw JSON authority contains {\"split_seed\":2026,\"split_seed\":3407}. The parser silently keeps the second value and the verifier does not reject duplicate object keys."),
    BenchmarkCase("case-007", "The verifier prepends the working directory to sys.path before importing neuros.research.verifier. The artifact bundle is allowed to contain its own neuros/research/verifier.py."),
    BenchmarkCase("case-008", "Repository pytest configuration declares asyncio_default_fixture_loop_scope, but the dedicated CI job installs only pytest. It never installs pytest-asyncio before invoking pytest."),
    BenchmarkCase("case-009", "The execution plan expects 810 terminal shards. The ledger contains 809 distinct terminal shard identities, but settlement is nevertheless marked complete."),
    BenchmarkCase("case-010", "Two lease records have different ordinals but the same shard_spec_sha256, participant, session, split seed, model seed, and method. Both are counted toward completeness."),
    BenchmarkCase("case-011", "The frozen plan specifies split_seed=2026. The worker CLI is invoked with --split-seed 2027, and the produced score is accepted into the planned evaluation."),
    BenchmarkCase("case-012", "After each completed shard, the scheduler reads validation balanced accuracy and sends later shards to whichever provider has produced the highest accuracy so far."),
    BenchmarkCase("case-013", "The feature builder appends the true class indicator as an auxiliary channel during both training and evaluation because downstream code expects a fixed channel count."),
    BenchmarkCase("case-014", "PCA and z-score parameters are fit once on the concatenation of train and held-out participants. The transformed held-out rows are then used for final evaluation."),
    BenchmarkCase("case-015", "A result artifact records metric values and model name, but omits source revision, input-data hash, preprocessing configuration, and model-state hash."),
    BenchmarkCase("case-016", "Evaluation order is created with random.shuffle(cases) and stochastic test-time crops are enabled. No seed or RNG state is recorded anywhere in the run manifest."),
    BenchmarkCase("case-017", "The claimed effect is participant-level, but a p-value is computed over 12,000 windows as if all windows were independent observations. Participants contribute unequal numbers of windows."),
    BenchmarkCase("case-018", "Fleet authority sets max_attempts_per_lease=2. A lease fails twice, receives a third claim, succeeds on attempt 3, and is accepted into settlement."),
    BenchmarkCase("case-019", "A worker writes result.json. Settlement accepts any file at that path if it exists; neither the file bytes nor its lease identity are hashed into the receipt."),
    BenchmarkCase("case-020", "A synthetic benchmark with generated EEG shows a decoder above chance. The report concludes that the decoder is clinically validated for unseen patients with real neurological disease."),
    BenchmarkCase("case-021", "The candidate commit and every cited check receipt bind the same full source SHA. The reviewer verifies that the receipts were generated after the candidate stopped changing."),
    BenchmarkCase("case-022", "Participants are assigned to train or test as whole groups before any learned transform is fit. No participant contributes samples to both partitions."),
    BenchmarkCase("case-023", "A scaler and PCA model are fit using training participants only, serialized with hashes, and applied without refitting to held-out participants."),
    BenchmarkCase("case-024", "Provider selection uses accelerator type, queue availability, price ceiling, and environment compatibility. Scientific metrics are not visible to the scheduler."),
    BenchmarkCase("case-025", "A lease allows two attempts. Attempt 1 fails with a trusted infrastructure-preemption class; attempt 2 succeeds. The settlement contains exactly one terminal artifact for every expected lease."),
    BenchmarkCase("case-026", "The review command executes python -m verifier from an extracted artifact directory. That directory contains verifier.py with the same module name as the trusted verifier, and import origin is never checked."),
    BenchmarkCase("case-027", "The PR is at commit ccc..., while its evidence receipt binds commit bbb.... In addition, the runtime reports numpy 2.2 although the frozen environment authority binds numpy 2.1."),
    BenchmarkCase("case-028", "The roster expects identities A, B, and C. Records A and B are present, then B is repeated under a new ordinal. The completeness counter reports three leases."),
    BenchmarkCase("case-029", "A global normalization transform is fit before the participant split using every row, and one generated feature is the per-row target encoded as an integer."),
)
_CASE_BY_ID = {case.case_id: case for case in _CASES}


def benchmark_cases() -> tuple[BenchmarkCase, ...]:
    """Return the frozen reviewer-visible corpus without scorer labels."""
    return _CASES


def benchmark_case(case_id: str) -> BenchmarkCase:
    normalized = require_nonempty(case_id, name="case_id")
    try:
        return _CASE_BY_ID[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark case {normalized!r}") from exc


def benchmark_public_manifest() -> dict[str, Any]:
    """Return public corpus metadata without importing scorer-owned ground truth."""
    return {
        "schema": BENCHMARK_SCHEMA,
        "corpus_sha256": BENCHMARK_CORPUS_SHA256,
        "case_count": len(_CASES),
        "taxonomy": [defect.to_dict() for defect in DEFECT_TAXONOMY],
        "cases": [case.to_public_dict() for case in _CASES],
        "ground_truth_included": False,
    }


def build_benchmark_task(
    case: BenchmarkCase | str,
    *,
    repository: str,
    source_revision: str,
) -> SealedSwarmTask:
    """Build a reviewer task without importing the scorer or answer key."""
    if isinstance(case, str):
        selected = benchmark_case(case)
    else:
        selected = benchmark_case(case.case_id)
        if selected != case:
            raise ValueError("benchmark case payload does not match the frozen corpus")
    revision = require_sha256(source_revision, name="source_revision")
    repo = require_nonempty(repository, name="repository")
    return SealedSwarmTask(
        repository=repo,
        source_revision=revision,
        objective=(
            "Identify every defect present in the stimulus. Emit zero findings for a clean case. "
            "Each finding_id must be one exact defect_id from the supplied taxonomy."
        ),
        claim_boundary=(
            "Benchmark classification only. Model output cannot authorize code, provider execution, "
            "scientific promotion, Kumar2024 execution, or ORION comparison."
        ),
        forbidden_actions=(
            "merge code",
            "authorize provider execution",
            "promote scientific claims",
            "change benchmark ground truth",
        ),
        public_context={
            "benchmark_schema": BENCHMARK_SCHEMA,
            "corpus_sha256": BENCHMARK_CORPUS_SHA256,
            "case": selected.to_public_dict(),
            "defect_taxonomy": [defect.to_dict() for defect in DEFECT_TAXONOMY],
            "scoring_rule": (
                "Report only present defects. Unknown finding IDs are invalid. False positives on "
                "clean controls reduce precision."
            ),
        },
    )
