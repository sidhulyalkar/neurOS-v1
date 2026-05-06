#!/usr/bin/env python3
"""
Example of dataset triage for astrocyte reanalysis potential.

This script demonstrates how to score datasets for astrocyte analysis suitability.
"""

from neuros_astro.metadata.dataset_scoring import score_dataset_metadata


def main():
    """Run dataset triage examples."""
    print("=" * 70)
    print("neuros-astro: Dataset Triage Examples")
    print("=" * 70)

    # Example 1: High-value astrocyte dataset
    print("\n[Example 1] High-value astrocyte imaging dataset")
    print("-" * 70)

    high_value_metadata = {
        "session_id": "high_value_example",
        "description": (
            "Two-photon calcium imaging of GFAP-labeled cortical astrocytes "
            "using GCaMP6f indicator. Includes raw imaging data, behavior tracking, "
            "and stimulus timing."
        ),
        "imaging_modality": "two-photon",
        "indicator": "GCaMP6f",
        "promoter": "GFAP",
        "has_raw_movie": True,
        "has_masks": True,
        "has_behavior": True,
    }

    result1 = score_dataset_metadata(high_value_metadata)

    print(f"Session ID: {result1.session_id}")
    print(f"Astro Score: {result1.astro_reanalysis_score:.2f}")
    print(f"Matched astro terms: {', '.join(result1.matched_astro_terms)}")
    print(f"Matched calcium terms: {', '.join(result1.matched_calcium_terms)}")
    print(f"Matched modality terms: {', '.join(result1.matched_modality_terms)}")
    print(f"Recommended step: {result1.recommended_next_step}")

    # Example 2: Neuron-only dataset
    print("\n[Example 2] Neuron-only electrophysiology dataset")
    print("-" * 70)

    neuron_only_metadata = {
        "session_id": "neuron_only_example",
        "description": (
            "Neuropixels recording from visual cortex during passive viewing. "
            "Contains spike times, LFP, and stimulus information."
        ),
        "has_ephys": True,
    }

    result2 = score_dataset_metadata(neuron_only_metadata)

    print(f"Session ID: {result2.session_id}")
    print(f"Astro Score: {result2.astro_reanalysis_score:.2f}")
    print(f"Matched astro terms: {', '.join(result2.matched_astro_terms) or 'None'}")
    print(f"Recommended step: {result2.recommended_next_step}")
    if result2.warnings:
        print(f"Warnings: {'; '.join(result2.warnings)}")

    # Example 3: Calcium imaging without astrocyte labels
    print("\n[Example 3] Calcium imaging without explicit astrocyte labels")
    print("-" * 70)

    ambiguous_metadata = {
        "session_id": "ambiguous_example",
        "description": (
            "Widefield calcium imaging with GCaMP in mouse cortex during "
            "behavioral tasks. Cell type not specified."
        ),
        "imaging_modality": "widefield",
        "indicator": "GCaMP",
        "has_behavior": True,
    }

    result3 = score_dataset_metadata(ambiguous_metadata)

    print(f"Session ID: {result3.session_id}")
    print(f"Astro Score: {result3.astro_reanalysis_score:.2f}")
    print(f"Matched calcium terms: {', '.join(result3.matched_calcium_terms)}")
    print(f"Matched modality terms: {', '.join(result3.matched_modality_terms)}")
    print(f"Recommended step: {result3.recommended_next_step}")
    if result3.warnings:
        print(f"Warnings: {'; '.join(result3.warnings)}")

    print("\n" + "=" * 70)
    print("Interpretation:")
    print("  - High scores (>0.7): Strong candidate for astro reanalysis")
    print("  - Medium scores (0.4-0.7): Worth manual inspection")
    print("  - Low scores (<0.4): Unlikely to contain astro signals")
    print("=" * 70)


if __name__ == "__main__":
    main()
