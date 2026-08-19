# Mechanistic interpretability learning track

This maintained track teaches mechanistic interpretability as experimental science rather than as a gallery of attractive activations.

Recommended order:

1. `00_what_counts_as_mechanistic_evidence.ipynb`
2. `01_ground_truth_causal_network.ipynb`
3. `02_activation_and_path_patching.ipynb`
4. `03_orion_token_causal_audit.ipynb`
5. `04_orion_representation_stability.ipynb`
6. `05_shared_neural_computation_study.ipynb`
7. `06_neurofm_mechanism_emergence.ipynb`
8. `07_circuit_faithfulness.ipynb`
9. `08_held_out_evidence_packs.ipynb`
10. `09_factorial_architecture_tokenizer.ipynb`
11. `10_causal_feature_correspondence.ipynb`
12. `11_hierarchical_replication.ipynb`
13. `12_reproducible_evidence_closure.ipynb`

The maintained research ladder is:

```text
question
→ discovery hypothesis
→ held-out causal intervention
→ necessity + sufficiency + matched controls
→ evidence pack
→ estimable architecture × tokenizer contrast
→ held-out causal feature correspondence
→ claim-aware hierarchical replication
→ dose response / manifold robustness
→ independent execution reproduction
→ explicit empirical evidence-closure status
```

## v1 release-evidence execution

Valid notebook JSON is not enough for a stable teaching surface. CI now executes the CPU-safe subset:

- 00 evidence definitions;
- 01 known causal network;
- 07 circuit faithfulness;
- 08 held-out evidence-pack rejection;
- 11 hierarchical replication/pseudoreplication;
- 12 independent reproduction/evidence closure.

Run the same suite locally:

```bash
python scripts/bootstrap.py --profile kernel --test-tools
pip install -e "packages/neuros-mechint[dev,notebooks]"
python packages/neuros-mechint/scripts/execute_cpu_tutorials.py
```

ORION/NeuroFM notebooks remain integration-tested at the API level in routine PR CI and should be executed in the environment that owns the corresponding data/model artifacts.

## Non-negotiable study rules

For real studies, align interventions to the same semantic event, freeze discovery/validation partitions before candidate selection, match task performance and model/token budgets, preserve missing/non-estimable cells, require held-out intervention evidence for causal correspondence, declare the independent replication unit before aggregation, and keep negative results.

A model-seed claim requires independent model seeds. A subject claim requires independent subjects. A cross-dataset “shared meaning” claim requires both causal and semantic alignment evidence. Raw latent feature indices are never assumed to correspond across independently trained models.
