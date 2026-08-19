# Changelog

All notable changes to the neuros-mechint package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project follows Semantic Versioning.

## [Unreleased]

## [1.0.0] - 2026-08-19

### Added
- Manifest schema v3 with deterministic `scientific_fingerprint` and execution-specific `run_hash`, keeping scientific identity distinct from runtime/host provenance.
- `ArtifactSchemaSpec`, `schema_catalog(...)`, `migrate_manifest_payload(...)`, `migrate_artifact_envelope(...)`, and validation helpers for the frozen v1 evidence contract.
- Frozen artifact-family contracts for held-out evidence packs, factorial mechanism studies, causal feature correspondence, hierarchical replication, and intervention dose response.
- Backwards migration from manifest schema v2 to v3 without allowing timestamped benchmark metadata to enter the scientific fingerprint.
- Backwards migration of supported pre-v1 artifact envelopes by validating the existing family schema/full-content hash and attaching canonical v1 contract metadata without rewriting the scientific result.
- `ReproductionSnapshot`, `ReproductionMetricTolerance`, `ReproductionSpec`, `ReproductionMetricComparison`, `ReproductionResult`, and `assess_independent_reproduction(...)` for tolerance-aware independent execution checks.
- Reproduction controls requiring, when preregistered, the same scientific fingerprint, a distinct run hash, a distinct execution identity, the same qualitative scientific decision, and bounded numerical drift.
- Self-checking v1 dose-response artifacts via `serialize_dose_response(...)`, `write_dose_response_artifact(...)`, `read_dose_response_artifact(...)`, and `neuros-mechint verify-dose-response-artifact`.
- `EvidenceRequirementState`, `EvidenceClosureRequirement`, `V1EvidenceStatus`, and `default_v1_evidence_status()` for machine-readable separation of software-contract readiness from empirical evidence completion.
- `neuros-mechint schemas --json` and `neuros-mechint release-status --json`.
- `neuros-mechint v1-ground-truth --json`, the ninth maintained synthetic scientific gate.
- The v1 ground-truth gate verifies manifest migration, legacy artifact migration, schema completeness, successful tolerance-bounded independent reproduction, duplicate-run rejection, qualitative decision-flip rejection, software-contract readiness, and empirical-overclaim rejection.
- Stable v1 method cards for versioned artifact schemas, independent artifact reproduction, and evidence-closure reporting.
- `tutorial-ci` optional dependencies for lightweight notebook execution without requiring full JupyterLab.
- `packages/neuros-mechint/scripts/execute_cpu_tutorials.py` and a dedicated CI job that executes the maintained CPU-safe evidence notebooks.
- `tutorials/mechint/12_reproducible_evidence_closure.ipynb`, demonstrating independent rerun semantics and the software-versus-empirical evidence boundary.
- `docs/V1_RELEASE_CONTRACT.md`, `docs/V1_EMPIRICAL_EVIDENCE_STATUS.md`, `docs/SCHEMA_MIGRATIONS.md`, `docs/V1_RELEASE_NOTES.md`, and `experiments/mechint/v1_evidence/README.md`.
- A concrete real-study landing zone for a matched Transformer × SSM crossed with Event × Relative-ISI design across at least three independent model-training seeds.

### Changed
- Promoted `neuros-mechint` from v0.9.0 to v1.0.0 and moved the package classifier from Alpha to Beta.
- Reframed v1 as a stable software/evidence contract rather than a claim that the corresponding real neuroscience experiments have already succeeded.
- The maintained scientific progression now ends in independent execution reproduction and explicit evidence closure after held-out causal correspondence, hierarchical replication, dose response, and manifold-aware controls.
- CI now runs nine scientific gates, Python 3.10-3.12 maintained tests and Ruff, ORION/NeuroFM integration, TransformerLens/NNsight/SAELens compatibility, repository-wide neurOS checks, and executed CPU evidence tutorials.
- The v0.9→v1 roadmap is now historical; outstanding empirical work is tracked separately and remains pending until immutable real-study artifact fingerprints exist.

### Fixed
- Corrected the v1 schema module to use `collections.abc.Mapping` and type-appropriate `TypeError` failures, resolving the final Ruff `UP035` and `TRY004` release-candidate failures.
- Prevented a timestamped/environment-sensitive `BenchmarkManifest` from being mistaken for the deterministic scientific identity of an experiment.
- Prevented a duplicate execution or a rerun with a flipped qualitative decision from satisfying the default independent-reproduction contract.
- Prevented software CI success from silently marking real neural evidence requirements complete.

### Scientific claim boundary
- v1.0 certifies the maintained framework contracts, migrations, falsification gates, artifact integrity checks, tutorial execution, and reproduction machinery; it does not certify a biological mechanism or a superior tokenizer/architecture without the corresponding real evidence artifacts.
- The repository intentionally reports real-model faithfulness, the matched Transformer × SSM × Event × Relative-ISI study, held-out causal correspondence, multi-seed correspondence replication, subject/session uncertainty, cross-context causal transfer, real dose response, stronger manifold-aware controls, independent artifact reproduction, and published negative real results as pending until actually executed.
- Same scientific fingerprint plus a successful reproduction decision does not prove organizational/statistical independence if two executions share undeclared upstream causes.
- Cross-dataset “shared meaning” still requires both causal and semantic alignment evidence; representation similarity alone remains insufficient.
- Donor/manifold provenance labels do not automatically establish that an intervention is in-distribution or biologically valid.

## [0.9.0] - 2026-08-19

### Added
- `ReplicationAxis`, `ReplicationCoordinates`, `ReplicationObservation`, `HierarchicalReplicationPolicy`, `HierarchicalReplicationSpec`, `IndependentUnitSummary`, `MetricReplicationEstimate`, `ReplicationDecision`, and `HierarchicalReplicationResult` for claim-aware higher-level mechanistic replication.
- Explicit scientific hierarchy support across dataset, model-training seed, checkpoint, feature dictionary, intervention projector, subject, session, and trial.
- Unit-balanced recursive aggregation so lower-level sample-count imbalance cannot give one model seed, subject, or other claim-level unit disproportionate weight.
- Deterministic hierarchical bootstrap confidence intervals that resample the declared scientific hierarchy rather than flattening correlated observations.
- Separate `estimable` and `replicated` decisions, preserving studies that are valid to estimate but fail because independent units disagree in sign, confidence intervals cross the null, or effects miss preregistered thresholds.
- Minimum independent-unit, estimable-fraction, sign-agreement, absolute-effect, direction, and confidence-interval gates.
- `observation_from_factorial_contrast(...)`, which carries v0.7 factorial effects into v0.9 without erasing non-estimable status or rejection reasons.
- `observation_from_correspondence(...)`, which carries v0.8 causal recovery, causal score, predictive transfer, random/shuffled control margins, source effect, target effect, source study fingerprint, and promotion metadata into a replication family.
- `InterventionManifoldKind` and `InterventionManifoldAssumption` for explicit zero, mean, empirical-donor, nearest-neighbor, quantile-matched, conditional, generative, causal-scrubbing-style, and custom replacement assumptions.
- Donor-pool provenance requirements for donor-based interventions and discovery-fit partition provenance for conditional/generative donors.
- `DoseResponseObservation`, `DoseResponsePolicy`, `DoseResponseSpec`, `DoseResponseUnitSummary`, `DoseResponseResult`, and `analyze_dose_response(...)` for preregistered graded intervention studies.
- Dose-response reporting for aggregate curves, oriented endpoint effects, within-unit monotonicity, normalized response area, and explicit rejection reasons.
- NumPy-1.24-compatible explicit trapezoidal integration for dose-response response area.
- `write_replication_artifact(...)`, `read_replication_artifact(...)`, and `neuros-mechint verify-replication-artifact` for self-checking hierarchical-replication artifacts.
- `neuros-mechint replication-ground-truth --json`, the eighth maintained synthetic scientific gate.
- The v0.9 ground-truth benchmark requires recovery of a positive mechanism across four independent model seeds, correct four-seed counting despite unequal lower-level sample counts, rejection of 300 strong trials from one seed as model-seed replication, rejection of a four-seed 50/50 sign-disagreement result, and recovery of a known monotonic five-dose curve.
- v0.9 method cards for Research claim-aware hierarchical replication, Integrated factorial→replication bridging, Research correspondence replication, and Research intervention dose response.
- `docs/HIERARCHICAL_REPLICATION.md`, `docs/ROADMAP_V0_9_TO_V1.md`, `tutorials/mechint/11_hierarchical_replication.ipynb`, and `experiments/mechint/replication_studies/README.md`.
- Maintained v0.9 tests for pseudoreplication rejection, unbalanced lower-level sample counts, sign heterogeneity, hierarchy-coordinate validation, deterministic bootstrap output, v0.7/v0.8 bridges, artifact integrity, dose-response rejection/recovery, and manifold donor provenance.

### Changed
- Promoted the package from v0.8.0 to v0.9.0. v0.8 established held-out causal feature correspondence; v0.9 establishes which scientifically independent levels actually replicate that evidence and quantifies higher-level uncertainty.
- Replication claims are now defined by an explicit claim axis. Extra trials, neurons, tokens, sessions, or perturbations improve lower-level precision but cannot manufacture additional model-seed, subject, dictionary, or dataset replicas.
- The maintained architecture now distinguishes the descriptive v0.7 `FactorialReplicationSummary` from the uncertainty-aware v0.9 `HierarchicalReplicationResult`.
- Negative and non-estimable source results remain in the replication analysis instead of filtering only to promoted/positive studies.
- Intervention replacement semantics now carry explicit manifold assumptions rather than implicitly treating zero, mean, empirical, and learned donors as equivalent.
- The active roadmap now begins at v0.9 and focuses the path to v1 on schema freeze/migrations, real matched architecture × tokenizer neural-data studies, independently replicated correspondence, cross-session/cross-dataset causal transfer, stronger manifold-aware controls, executed tutorials, and independent reproduction.
- CI now runs eight synthetic scientific gates while retaining Python 3.10-3.12, ORION/NeuroFM integration, focused Ruff, repository hygiene, and real-package TransformerLens/NNsight/SAELens import checks.

### Scientific claim boundary
- A passing v0.9 result supports only replication under the declared family, metric, hierarchy, independent-unit definition, null, direction, intervention protocol, and replication policy.
- Hundreds of trials from one model seed are not architecture/model-seed replication; many sessions from one subject are not subject-level replication.
- Hierarchical bootstrap uncertainty remains conditional on the declared nesting structure and can be unstable when very few higher-level independent units exist.
- Replication across model seeds does not automatically establish subject-, dataset-, species-, tokenizer-, dictionary-, or projector-level transfer.
- A monotonic intervention dose response is supporting evidence rather than unique proof of a mechanism.
- A donor labeled empirical, conditional, generative, or otherwise manifold-aware is not automatically valid; donor-pool semantics, fit partition, and projector meaning remain part of the scientific claim.
- A failed v0.8 correspondence can be an estimable negative replica, while a non-estimable v0.7 contrast remains non-estimable and cannot be repaired by hierarchical aggregation.

## [0.8.0] - 2026-08-18

### Added
- `FeatureSpaceIdentity`, `FeaturePairExample`, `FeatureCorrespondenceSpec`, `FeatureCorrespondenceCandidate`, `FeatureCorrespondencePolicy`, `CorrespondenceValidationMetrics`, `CorrespondencePromotionDecision`, and `FeatureCorrespondenceResult` for discovery-frozen, held-out cross-representation correspondence studies.
- Explicit one-to-one, one-to-many, and subspace correspondence shapes instead of assuming raw latent indices have portable meaning.
- Semantic `example_id`, `semantic_trial_id`, discovery/validation split, and partition identities so renamed scientific trials cannot leak across correspondence discovery and validation.
- Explicit declaration of every source/target context axis that changes across model, revision, architecture, tokenizer, dataset, session, subject, and checkpoint; undeclared differences fail construction.
- Discovery-only ridge-linear source→target maps with frozen coefficients, intercepts, deterministic scientific identity, and rank-tolerant least-squares fitting for singular or collinear subspaces.
- Separate representation evidence for activation correlation, linear CKA geometry, semantic-label overlap, discovery predictive R², held-out predictive R², and discovery-to-validation degradation.
- `CausalSubstitutionEvaluator` and `CausalSubstitutionMetrics` for paired held-out source ablation, target ablation, and mapped target substitution.
- Causal credit that requires both source-feature and target-feature intervention relevance before mapped recovery can count as correspondence evidence.
- Shuffled semantic-trial donor controls that preserve the candidate feature family while breaking the scientific source/target pairing.
- Same-cardinality random-source feature controls where every control receives its own discovery-only mapping to the same target feature set before held-out causal evaluation.
- Scalable random-control sampling that avoids enumerating combinatorial feature subsets for SAE-scale feature universes.
- `TensorFeatureProjector`, `AdapterFeatureSpaceView`, `AdapterPairedExampleSpec`, `AdapterCausalSubstitutionEvaluator`, and `run_adapter_feature_correspondence_study(...)` for real ModelAdapter capture, ablation, substitution, metric evaluation, and model-state mutation guards.
- `FactorialCorrespondenceOrigin` and `factorial_origin_from_report(...)`, allowing v0.8 provenance to link only to an estimable v0.7 factorial contrast.
- `write_correspondence_artifact(...)`, `read_correspondence_artifact(...)`, and `neuros-mechint verify-correspondence-artifact` for self-checking correspondence artifacts that omit raw model inputs and raw activation arrays.
- `neuros-mechint correspondence-ground-truth --json`, a synthetic gate containing a true causal correspondence and a nearly perfectly predictive, semantically matched but causally unused decoy.
- The correspondence gate requires true held-out recovery, shuffled-pair separation, random-source separation, and rejection of the high-similarity decoy because its source-ablation effect is zero.
- v0.8 method cards separating Stable correspondence design/bookkeeping, Research held-out causal substitution, Integrated ModelAdapter execution, and Integrated v0.7→v0.8 provenance linking.
- `docs/CAUSAL_FEATURE_CORRESPONDENCE.md`, `docs/ROADMAP_V0_8_TO_V1.md`, `tutorials/mechint/10_causal_feature_correspondence.ipynb`, and `experiments/mechint/correspondence_studies/README.md`.
- Maintained tests for discovery isolation, semantic-trial leakage, undeclared context differences, one-to-many maps, subspace maps, rank-deficient zero-ridge fitting, 10,000-feature control sampling, artifact integrity, real PyTorch feature substitution, and factorial provenance rejection.

### Changed
- Promoted the package from v0.7.0 to v0.8.0. v0.7 established estimable architecture × tokenizer comparisons; v0.8 adds the held-out intervention layer needed to ask whether apparently aligned features actually transfer causal contribution.
- Representation similarity, geometric alignment, semantic agreement, predictive transfer, intervention-effect agreement, and causal substitution remain separate evidence objects instead of being collapsed into a universal correspondence score.
- The default tensor correspondence integration now requires an explicit feature axis and documents that averaging non-feature axes supports aggregate channel claims only; temporal/event-specific claims require a projector that preserves those coordinates.
- Random-source controls are fit fairly under the same discovery budget while scaling to large feature spaces without materializing all combinations.
- Linear mapping fit no longer assumes an invertible Gram matrix when `ridge_alpha=0`; rank-deficient subspaces use least-squares solutions.
- The CI matrix now runs seven scientific synthetic gates while retaining Python 3.10-3.12, ORION/NeuroFM integration, focused Ruff, repository hygiene, and real-package TransformerLens/NNsight/SAELens import checks.
- The active roadmap now begins at v0.8 and prioritizes hierarchical uncertainty, multi-seed/session/subject replication, dose-response interventions, and stronger in-manifold controls in v0.9.

### Scientific claim boundary
- A passing v0.8 result establishes only conditional causal substitutability under the frozen source/target models, revisions, feature spaces, projector, metric, discovery mapping, semantic trial pairing, intervention family, and matched-control policy.
- High activation correlation, geometric similarity, semantic-label overlap, or predictive R² does not establish causal correspondence without source/target intervention relevance and held-out substitution evidence.
- Successful substitution does not establish feature uniqueness, biological homology, universal semantic identity, equality of raw latent indices, in-manifold intervention validity, cross-dataset transfer, or replication across model/dictionary seeds or subjects.
- An estimable v0.7 factorial interaction may nominate a v0.8 correspondence study but cannot identify corresponding features or bypass independent discovery/validation.
- Similarity-without-causality, failed substitution, and failed matched controls are first-class scientific results and should be retained.

## [0.7.0] - 2026-08-18

### Added
- `FactorialMechanismSpec`, `FactorialCellSpec`, `FactorialCellOutcome`, `FactorialContrastSpec`, `FactorialContrastResult`, `FactorialReplicationSummary`, and `FactorialMechanismReport` for preregistered architecture × tokenizer mechanism studies.
- `MatchedCovariate` and `FactorialAnalysisPolicy` for executable matching of token budget, temporal resolution, downstream capacity, training compute, checkpoint maturity, task performance, and study-specific nuisance dimensions.
- Explicit semantic `discovery_partition_id` and `validation_partition_id` fields so different tokenizers can be compared on the same underlying neural trials without pretending their raw token tensors should match.
- Evidence-protocol fingerprints over discovery method, metric, target universe, intervention baselines, faithfulness policy, evidence-pack policy, and random-control budget.
- Preregistered architecture, tokenizer, checkpoint, and architecture × tokenizer difference-in-differences contrasts.
- Explicit estimability decisions that preserve missing cells and reject confounded comparisons instead of silently dropping or averaging them.
- Optional causal effect-map stability and target-wise interaction maps when intervention targets are meaningfully aligned.
- `preregister_2x2_contrasts(...)` for materializing the five primary contrasts in a matched 2 × 2 architecture/tokenizer slice.
- Cross-session replication groups with estimable-count, session coverage, sign agreement, median effects, and a separate Research-maturity replication-readiness flag.
- `FactorialEvidenceCellInput` and `run_factorial_evidence_study(...)`, bridging completed v0.6 evidence packs plus optional v0.3/v0.4 causal maps into the factorial layer while validating revisions, partitions, protocol, checkpoint metadata, and matched covariates.
- `write_factorial_artifact(...)`, `read_factorial_artifact(...)`, and `neuros-mechint verify-factorial-artifact` for self-checking comparative-study artifacts.
- `neuros-mechint factorial-ground-truth --json`, a synthetic gate with a known `-0.5` architecture × tokenizer interaction replicated across two sessions.
- Negative factorial controls requiring the same gate to reject a token-budget-confounded tokenizer comparison and a 2 × 2 interaction with an explicitly missing cell.
- v0.7 method cards for Stable factorial design/contrast machinery, Integrated evidence-pack bridging, and Research cross-session replication.
- `docs/FACTORIAL_MECHANISM_STUDIES.md`, `docs/ROADMAP_V0_7_TO_V1.md`, `tutorials/mechint/09_factorial_architecture_tokenizer.ipynb`, and `experiments/mechint/factorial_studies/README.md`.
- Maintained tests for known interactions, confound rejection, missing-cell rejection, task-performance-matched checkpoint contrasts, protocol mismatch, artifact integrity, and an end-to-end real `EvidencePackResult` → factorial contrast path.

### Changed
- Promoted the package from v0.6.0 to v0.7.0. v0.6 established trustworthy held-out cell-level evidence; v0.7 adds the matched comparative layer needed to ask whether architecture or tokenization changes the learned causal mechanism.
- Comparative analysis now treats **estimability before effect size** as a first-class scientific rule.
- Task performance, circuit faithfulness, causal-map shape, candidate size, random-control performance, and intervention-baseline sensitivity remain separate outcomes instead of being collapsed into a universal tokenizer score.
- The CI matrix now runs six scientific synthetic gates while retaining Python 3.10-3.12, ORION/NeuroFM integration, focused Ruff, repository hygiene, and real-package TransformerLens/NNsight/SAELens import checks.
- The active roadmap now begins at v0.7 and prioritizes causal feature correspondence in v0.8, followed by hierarchical multi-seed/session/subject uncertainty and replication in v0.9.

### Scientific claim boundary
- An estimable factorial effect is conditional on the declared grid, source evidence packs, semantic partitions, matched covariates, target universe, task-performance tolerance, and evidence protocol.
- An architecture × tokenizer interaction does not establish a universal tokenizer advantage or identify why the interaction exists.
- Effect-map similarity or difference is not substituted for held-out necessity/sufficiency; the two evidence objects remain separate.
- Cross-session replication does not substitute for multiple subjects, datasets, or independent model-training seeds.
- A non-estimable contrast is a valid scientific result and should be retained with its rejection reasons.
- v0.7 does not establish causal feature correspondence across independently trained representations; that is the explicit v0.8 target.

## [0.6.0] - 2026-08-18

### Added
- `EvidenceSplit`, `EvidenceExample`, `EvidencePackSpec`, `EvidencePackPolicy`, `EvidenceCaseResult`, `EvidenceAggregate`, `EvidencePromotionDecision`, `EvidenceTelemetry`, and `EvidencePackResult` for frozen discovery-versus-validation circuit studies.
- `run_adapter_evidence_pack(...)`, which exposes only discovery examples to candidate selection, freezes candidate state before held-out intervention, and records explicit promotion/rejection reasons.
- Deterministic input-content hashing with rejection of duplicate inputs anywhere in an evidence pack, preventing renamed discovery examples from reappearing in validation.
- `fit_discovery_mean_references(...)`, which estimates per-target mean-ablation donors exclusively from discovery activations and freezes them for validation.
- Model-state mutation guards before discovery, after discovery/donor fitting, and after evidence evaluation when the adapter exposes deterministic fingerprint payloads.
- `discover_ablation_effect_candidate(...)`, a Research-maturity discovery-only candidate generator based on single-target zero-ablation effects.
- `discover_activation_magnitude_candidate(...)`, a same-cardinality non-causal baseline fitted on discovery examples and evaluated on identical held-out interventions.
- Evidence-pack aggregation with discovery/validation pass rates, joint-faithfulness summaries, invalid-case counts, and deterministic 95% bootstrap intervals.
- Example-paired bootstrap resampling so multiple intervention baselines on the same example are not treated as independent observations.
- Explicit invalid normalization records when the all-target/null span does not define the intended faithfulness comparison.
- Promotion policy controls for held-out sample count, pass rate, median joint faithfulness, discovery-to-validation degradation, invalid cases, intervention-family diversity, and performance relative to the same-size magnitude baseline.
- `publication_ready` and `publication_issues`, separating reproducibility/pinning status from scientific promotion so negative results can remain first-class artifacts.
- `write_evidence_pack_artifact(...)` and `read_evidence_pack_artifact(...)` for self-checking JSON artifacts that exclude raw inputs while preserving scientific results and provenance.
- Artifact integrity hashes, deterministic study fingerprints, run hashes, package versions, wall time, peak Python memory, and peak CUDA memory where available.
- `DiscoveryShiftMLP` and `neuros-mechint evidence-pack-generalization-ground-truth --json`, a negative scientific gate in which an in-sample-discovered circuit must be rejected on a known held-out mechanism shift.
- `neuros-mechint evidence-recipes` with maintained TransformerLens, NNsight, SAELens, and circuit-tracer real-model starting points that require immutable revision pinning before publication.
- `neuros-mechint verify-evidence-artifact` for schema/hash verification after copying or publishing an artifact.
- v0.6 method cards separating Stable evidence-pack bookkeeping, Stable magnitude baselines, Research candidate discovery, and Integrated external-model recipes.
- `docs/REAL_MODEL_EVIDENCE_PACKS.md` and `docs/ROADMAP_V0_6_TO_V1.md`.
- Maintained tests for discovery leakage, known held-out failure, duplicate-content leakage, frozen donor estimation, invalid normalization, revision readiness, artifact round-trip, and artifact tamper detection.

### Changed
- Promoted the package from v0.5.0 to v0.6.0. v0.5 established ecosystem-native adapters and single-input quantitative faithfulness; v0.6 establishes held-out study semantics and reproducible evidence artifacts.
- Mean module ablation no longer averages only dimension zero. A mean replacement fills the audited activation with an explicit scalar donor statistic; evidence-pack studies use donors fitted exclusively on discovery data.
- Faithfulness subset evaluations remain memoized, and v0.6 evidence packs preserve normalization failures instead of crashing an entire multi-example study or silently reversing the interpretation.
- The CI matrix now runs five scientific gates, including the known-overfit held-out rejection test, while retaining Python 3.10-3.12 coverage, ORION/NeuroFM integration, focused Ruff, repository hygiene, and real-package TransformerLens/NNsight/SAELens solver checks.
- The active roadmap now starts at v0.6 and makes factorial architecture x tokenizer evidence experiments the v0.7 focus.

### Fixed
- Resolved the final v0.5 dedicated-workflow Ruff failures in adapter import/export ordering and SAELens set construction.
- Prevented single-batch transformer activations from making the historical dim-0 mean ablation an accidental no-op.
- Prevented held-out examples from adapting their own mean-ablation donor statistics.
- Prevented exact duplicate input content from crossing discovery/validation boundaries under different IDs.
- Prevented candidate-discovery callbacks from silently mutating model parameters when a deterministic model fingerprint is available.
- Prevented baseline-level perturbations on one example from being counted as independent bootstrap samples.

### Scientific claim boundary
- A passing v0.6 evidence pack is conditional on the frozen model/data revisions, metric, target universe, discovery procedure, content-distinct held-out split, intervention families, and promotion policy.
- A held-out pass does not establish circuit uniqueness, in-manifold intervention validity, cross-dataset transfer, cross-seed stability, biological homology, or cross-model feature correspondence.
- Candidate discovery remains a separate scientific layer from held-out evidence. A Research discovery heuristic does not become Stable merely because the evidence-pack bookkeeping is Stable.
- A rejected candidate is a valid scientific result and should still be retained and published when its revisions/provenance are reproducible.
- External-model recipes are maintained execution configurations, not measured evidence.

## [0.5.0] - 2026-08-18

### Added
- `TransformerLensAdapter` for native `run_with_cache(...)` and `run_with_hooks(...)` activation capture/replacement across TransformerLens 3 hook surfaces and TransformerBridge-compatible objects.
- `NNsightAdapter` and `NNsightTarget` for trace-time `.output`, `.save()`, activation assignment, forward-order path sorting, and explicit tuple-output selectors such as `transformer.h.0::0`.
- `SAELensFeatureAdapter` and `SAEReconstructionAudit` for duck-typed SAE `encode()`/`decode()` workflows with reconstruction-gap accounting before feature intervention claims.
- `CircuitTracerAdapter` and `AttributionGraphSummary` for normalizing active-feature identities and probability-weighted direct feature-to-logit attribution edges without representing attribution as causal evidence.
- `CircuitCandidate`, `FaithfulnessPolicy`, `RandomCircuitControl`, and `CircuitFaithfulnessReport`.
- Generic circuit necessity/sufficiency evaluation with all-target/null normalization and equal-cardinality random controls.
- Joint faithfulness random percentile based on `min(sufficiency, necessity)` to avoid overvaluing random circuits that are necessary but insufficient in serial pathways.
- `evaluate_adapter_circuit_faithfulness(...)` for any `ModelAdapter`, allowing PyTorch, TransformerLens, and NNsight candidates to face the same quantitative benchmark.
- `evaluate_sae_feature_faithfulness(...)`, which performs feature-set faithfulness relative to the SAE reconstruction baseline and records the original/reconstruction metric gap.
- Known-circuit synthetic benchmark and `neuros-mechint circuit-faithfulness-ground-truth --json` scientific gate.
- Protocol-faithful CPU tests for TransformerLens, NNsight, SAELens, and circuit-tracer adapter contracts without model downloads.
- Real-package import/solver CI jobs for the supported TransformerLens 3.x, NNsight 0.7.x, and SAELens 6.x optional extras.
- `docs/ECOSYSTEM_ADAPTERS.md`, `docs/CIRCUIT_FAITHFULNESS.md`, and a post-v0.5 roadmap focused on real evidence packs and replication.
- `07_circuit_faithfulness.ipynb`, a maintained CPU tutorial demonstrating necessity, sufficiency, and same-size random controls on a known mechanism.

### Changed
- Promoted the package from v0.4.0 to v0.5.0. v0.4 established the neural-foundation-model/tokenizer/checkpoint laboratory; v0.5 adds external ecosystem interoperability and a shared quantitative promotion gate for nominated circuits.
- TransformerLens, NNsight, SAELens, and circuit-tracer registry entries move from `planned` to `integrated` while preserving optional imports.
- Package extras now provide bounded TransformerLens 3.x, NNsight 0.7.x, and SAELens 6.x dependency families. circuit-tracer remains upstream-installed rather than an unversioned Git dependency.
- Method cards now separate Stable circuit-faithfulness mathematics from Integrated external adapter compatibility.
- CI now runs four independent scientific synthetic gates and ecosystem dependency smoke checks in addition to Python 3.10-3.12, ORION/NeuroFM integration, tests, lint, and repository hygiene.
- The active roadmap now begins at v0.5 and prioritizes held-out real-model evidence packs, factorial architecture x tokenizer experiments, causal feature correspondence, uncertainty, and independent replication.

### Fixed
- Resolved the v0.4 dedicated evidence-workflow Ruff failures in NeuroFM import ordering and an unused ORION tokenizer-study import before extending the package.
- TransformerLens baseline forward execution now uses the common `run_with_hooks(..., fwd_hooks=[])` path rather than assuming the wrapped bridge is directly callable.
- NNsight tuple-valued outputs require an explicit selector instead of silently replacing a structured output with a tensor.
- SAE feature intervention reports no longer allow reconstruction error to masquerade as feature causality.

### Scientific claim boundary
- A circuit-tracer attribution graph nominates candidate features; it is not converted directly into `CausalEffectRecord`.
- Adapter maturity establishes API/contract compatibility, not correctness of a discovered circuit.
- A passing faithfulness report is conditional on the chosen input distribution, metric, audited target universe, and intervention family. It does not establish circuit uniqueness or in-distribution perturbations.
- SAE feature claims must report the reconstruction metric gap and should replicate across dictionaries/seeds before stronger interpretation.
- Real mechanism claims still require held-out examples, alternative interventions, multiple seeds/checkpoints, uncertainty estimates, and independent replication.

## [0.4.0] - 2026-08-18

### Added
- `CheckpointMechanismState`, `TargetEmergence`, `CheckpointSimilarity`, and `MechanismEmergenceReport` for longitudinal causal-map analysis across training checkpoints.
- `analyze_mechanism_emergence(...)` with explicit effect-magnitude, rank, sign, target-coverage, and consecutive-checkpoint thresholds.
- Known-transition checkpoint benchmark and `neuros-mechint mechanism-emergence-ground-truth --json` CI gate.
- Separate tokenizer-mechanism contracts: `TokenizerMechanismContext`, `TokenizerEffectRecord`, matched pair comparison, stability aggregation, and tokenizer-specific falsifiable hypotheses.
- `OrionTokenizerStudyContext` and `run_orion_tokenizer_study(...)` for event-relative causal comparison of neural tokenization schemes.
- Tokenizer-study scientific/run identities: deterministic `study_fingerprint`/`study_hash`, timestamped context manifest hashes, and `run_hash`.
- `ModelCall` for PyTorch models requiring multiple positional or keyword inputs such as an attention mask.
- `NeuroFMRepresentationProbe` and `NeuroFMProbeSpec` for extracting tensor-valued internal neural-foundation-model states into ORION `RepresentationBatch` contracts.
- Explicit compressed-state timestamp validation: temporal coordinates must be supplied when a captured component no longer matches input sequence length.
- `NeuroFMCheckpointContext`, `NeuroFMMechanismLabResult`, and `run_neurofm_mechanism_lab(...)` combining internal-state capture, event-aligned interventions, architecture comparison, and checkpoint emergence.
- v0.4 method cards for tokenizer studies, NeuroFM probes, mechanism emergence, and the NeuroFM mechanism laboratory.
- CPU-only maintained tests for tokenizer confound rejection, semantic token alignment, multi-argument model calls, internal-state capture, compressed timestamps, matched architecture contrasts, and checkpoint trajectories.
- `neurofm` optional dependency group for users who want the complete workspace integration.

### Changed
- Promoted the package from v0.3.0 to v0.4.0. v0.3 established comparative causal maps; v0.4 adds tokenizer, architecture, and training-time mechanism laboratories.
- `PyTorchAdapter.forward(...)` now accepts `ModelCall` while retaining the original single-input behavior.
- Tokenizer is represented as its own experimental axis instead of being overloaded into architecture metadata.
- Checkpoint-emergence analysis requires architecture, dataset, session, and subject to remain fixed so checkpoint is the sole varying scientific axis.
- The maintained CI matrix now validates v0.4 on Python 3.10-3.12, runs three scientific ground-truth gates, tests ORION tokenizer studies and the NeuroFM mechanism lab, and lints the new supported modules.
- The roadmap now begins at v0.4 and prioritizes external adapter faithfulness, real neural evidence packs, factorial architecture x tokenizer experiments, causal feature correspondence, multi-seed emergence, uncertainty, and replication.

### Scientific claim boundary
- The final observed checkpoint is a reference for mechanism-emergence analysis, not biological ground truth.
- A tokenizer-dependent causal profile makes tokenization a candidate source of computational bias; it does not establish that one tokenizer is superior without matched token budget, temporal resolution, downstream capacity, training budget, and held-out evidence.
- Capturing a NeuroFM hidden state establishes extraction, not causal relevance. Causal relevance still requires intervention experiments.
- Compressed latent indices are not treated as time unless explicit timestamps establish that correspondence.
- Architecture comparisons require isolated architecture contrasts, matched task definitions, and matched training maturity before they support architecture-specific interpretations.

## [0.3.0] - 2026-08-18

### Added
- Generic `CausalEffectRecord` and `MechanismContext` contracts for causal-map comparison across architectures, datasets, sessions, subjects, and checkpoints.
- Pairwise causal-map comparison with Pearson/Spearman agreement, sign agreement, top-k overlap, effect drift, target counts, and explicit shared-target fraction.
- Both descriptive `axis_stability` and one-factor-at-a-time `isolated_axis_stability` summaries.
- Architecture summaries covering task-score level/variance, causal-effect magnitude, matched-control ratio, causal concentration, and within-family stability.
- `HypothesisPolicy`, `MechanisticHypothesis`, and `SharedComputationAnalysis` for transparent, threshold-driven prioritization of falsifiable candidate hypotheses.
- Minimum shared-target coverage as an explicit hypothesis-policy requirement so high correlation on a small overlap cannot masquerade as strong mechanistic agreement.
- Candidate hypothesis families for architecture-invariant causal structure, architecture-specific implementations, context invariance, perturbation/distribution-shift sensitivity, and concentrated causal support.
- Synthetic shared-computation benchmark containing known shared and known architecture-specific causal maps.
- `neuros-mechint shared-computation-ground-truth --json` as a CI-enforced scientific gate for the comparison/hypothesis layer.
- ORION `orion_study` orchestration from `RepresentationBatch` through temporal interventions, canonical effect maps, architecture comparison, and hypotheses.
- Event-relative temporal canonicalization via per-context `alignment_origin_ns`, allowing sessions with unrelated absolute recording clocks to be compared against a common semantic event.
- `alignment_label` validation for explicitly labeled ORION contexts so incompatible semantic events fail loudly.
- Per-context ORION feature-group necessity audits with temporal-permutation matched controls.
- Separate reproducibility identities: deterministic `study_fingerprint`/`study_hash`, per-context timestamped manifest hashes, and run-specific `run_hash`.
- `05_shared_neural_computation_study.ipynb`, an end-to-end maintained tutorial spanning ORION representations through falsification-oriented hypotheses.
- Maintained-notebook JSON validity test.

### Changed
- Promoted the package from v0.2.0 to v0.3.0: v0.2 established the trustworthy causal experiment spine; v0.3 adds comparative mechanism science.
- `EffectMapStability` now reports left/right/union target counts and `shared_target_fraction` so sparse overlap remains visible.
- Hypothesis generation now uses one-factor-at-a-time comparisons where exactly one context axis changed, while broader multi-axis comparisons remain descriptive.
- Matched intervention and control maps are extracted separately to avoid duplicate-target ambiguity.
- Method cards now distinguish Stable causal-map statistics, Integrated ORION comparative studies, and the Research-maturity hypothesis engine.
- CI validates v0.3 on Python 3.10-3.12, runs both known-ground-truth scientific gates, exercises the ORION shared-computation study, validates maintained notebooks, and lints the supported package surface.
- CI now cancels superseded mech-int runs on the same ref to avoid wasting runner capacity on obsolete SHAs.
- Documentation treats generated hypotheses as experiment queues requiring held-out validation, not automatic discoveries.

### Fixed
- Removed an explicit `return None` from the receiver-capture forward hook in `circuits/path_patching.py`, resolving Ruff `RET501` and `PLR1711` failures while preserving hook semantics.
- Bound receiver/cache state in the path-patching hook defaults so loop closures do not capture the wrong sender/receiver state.
- Cross-session comparisons no longer depend on matching absolute nanosecond clocks when a common semantic alignment origin is supplied.
- Corrected the provenance model so timestamped `BenchmarkManifest` data is not mislabeled as a deterministic scientific-study identity.

### Scientific claim boundary
- Causal-map agreement means agreement of the measured intervention profile under a specified experiment. It is not proof of an identical biological mechanism.
- Raw latent feature indices are not assumed to correspond across independently trained models without a separate alignment experiment.
- Pairwise comparisons preserve every changed context axis, and hypothesis generation preferentially uses matched one-axis-only pairs to reduce confounding.
- Shared-computation hypotheses require both effect agreement and substantial intervention-target overlap.

## [0.2.0] - 2026-08-18

### Added
- Stable causal experiment kernel with typed counterfactual pairs, interventions, scalar metrics, results, controls, and manifests.
- `ModelAdapter` boundary and default `PyTorchAdapter` for module-output tracing and replacement.
- Repository-aligned evidence tiers separate from method maturity.
- `ExperimentManifest` integration with `neuros.quality.BenchmarkManifest` plus full-content model/data hashing.
- Ground-truth localization benchmark with known causal and nuisance pathways, precision/recall/AP, and separation metrics.
- Framework-agnostic input causal audits for tokens, signals, and latent representations.
- ORION `NeuroTokenBatch` causal interventions for time windows, token types, side features, and deterministic shuffle controls.
- ORION `RepresentationBatch` causal interventions for latent time windows and feature dimensions with matched shuffle controls.
- Cross-context causal effect-map stability metrics covering Pearson/Spearman agreement, sign agreement, top-k overlap, and effect drift.
- `neuros-mechint` CLI evidence surfaces for method cards, evidence tiers, integration status, and the ground-truth benchmark.
- Maintained repository-level tutorial track under root `tutorials/`.
- GitHub Actions coverage for Python 3.10-3.12, ORION integration, lint, scientific smoke tests, and repository hygiene.

### Changed
- Repositioned `neuros-mechint` as the causal experiment/research layer above stable neurOS and ORION contracts rather than a parallel runtime or catch-all computational-neuroscience package.
- Python support now matches the refactored workspace (`>=3.10`).
- Common historical top-level APIs resolve lazily so optional research dependencies no longer make stable imports fragile.
- Activation patching and path patching are separate methods with different claim boundaries.
- The historical ACDC implementation is explicitly described as ACDC-inspired module pruning rather than canonical edge-level ACDC.
- Maintained teaching material follows repository policy and lives under root `tutorials/`; historical package notebooks remain research artifacts.
- CI uses CPU-only PyTorch and non-fail-fast Python matrices for lower-cost, more informative validation.

### Fixed
- Final circuit performance evaluates the retained circuit while rejected modules remain ablated.
- Provenance hashing distinguishes same-shaped arrays/tensors with different values and supports scalar and `bfloat16` tensors.
- The refactor branch is based directly on the current workspace architecture rather than carrying duplicated provenance/runtime infrastructure.

## Historical Phase 2 Expansion - 2025-11-04

This section records the package's exploratory pre-v0.2 expansion. Presence in this history does not imply current Stable method maturity or execution coverage.

### Added
- Thermodynamics of computation analyses including Landauer, NESS, fluctuation-theorem, energy-cascade, and Hamiltonian experiments.
- Advanced dynamics including neural ODE, slow-feature, flow-field, fixed-point, Koopman, Lyapunov, and manifold analyses.
- Meta-dynamics for training trajectories, representational drift, phase transitions, and feature emergence.
- Geometry/topology experiments including curvature, intrinsic dimensionality, persistent homology, and geodesics.
- Counterfactual/latent surgery, synthetic lesions, and exploratory do-calculus interfaces.
- Circuit research extensions including latent RNN extraction, mixed selectivity, activation maximization, circuit comparison, and motif detection.
- Biophysical experiments including spiking-neuron, plasticity, Dale-law, dendritic, and intervention models.
- Cross-species alignment, temporal dynamics, criticality, avalanche, and multifractal analyses.
- Historical notebooks 17-22 covering advanced biophysics, interventions, cross-species alignment, temporal dynamics, criticality, and multifractals.

### Historical status
These modules remain available as research artifacts. They are not promoted to the v0.6 Stable surface merely because code exists for them.

## [0.1.0] - 2025-10-27

### Added
- Sparse autoencoder research implementation with L1 sparsity and multi-layer training utilities.
- Historical circuit discovery and path-patching prototypes.
- CCA/RSA/Procrustes representation alignment.
- Fractal and 1/f analyses.
- Dynamical-systems analyses including Koopman, Lyapunov, and attractor methods.
- Initial result storage, pipeline, database, and notebooks 01-16.

## Development notes

The v1 software/evidence contract is complete. The next development priority is producing and independently reproducing the real neural evidence artifacts tracked in `docs/V1_EMPIRICAL_EVIDENCE_STATUS.md`, without promoting pending empirical requirements until the corresponding immutable fingerprints exist.
