#!/bin/bash
# Circuit Extraction & Mechanistic Interpretability Pipeline
# Run all experiments on your trained SAE

set -e  # Exit on error

echo "======================================================================"
echo "neurOS-v1 Circuit Extraction Pipeline"
echo "======================================================================"
echo ""
echo "Session: 754829445 (71% selective SAE features)"
echo "Expected total runtime: ~45-60 minutes"
echo ""

# Check conda environment
if [[ "$CONDA_DEFAULT_ENV" != "mechint_playground" ]] && [[ "$CONDA_DEFAULT_ENV" != "neurofm" ]]; then
    echo "⚠️  Please activate conda environment first:"
    echo "   conda activate mechint_playground"
    echo "   (or: conda activate neurofm)"
    exit 1
fi

echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Configuration
SESSION_ID=754829445
SAE_MODEL="sae_models/sae_session_${SESSION_ID}.pt"
ALLEN_CACHE="allen_validation_cache"
DEVICE="cpu"  # Change to "cuda" if GPU available

# Check if SAE model exists
if [[ ! -f "$SAE_MODEL" ]]; then
    echo "❌ SAE model not found: $SAE_MODEL"
    exit 1
fi

echo "======================================================================"
echo "Step 1: Feature Attribution Analysis (✅ COMPLETED)"
echo "======================================================================"
echo "Status: You already ran this! Results in results/circuits/"
echo "- 20 circuits extracted"
echo "- 71/92 neurons used (77%)"
echo "- 46 neurons reused across features"
echo ""

# Step 2: Ablation Study
echo "======================================================================"
echo "Step 2: Causal Circuit Validation via Ablation (~15-20 min)"
echo "======================================================================"
read -p "Run ablation study? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python experiments/circuit_extraction/ablation_study.py \
        --sae-model "$SAE_MODEL" \
        --attribution-results "results/circuits/attribution_results_session_${SESSION_ID}.json" \
        --session-id $SESSION_ID \
        --allen-cache "$ALLEN_CACHE" \
        --output-dir results/circuits/ablation \
        --top-features 10 \
        --device "$DEVICE"

    echo ""
    echo "✓ Ablation study complete! Check results/circuits/ablation/"
    echo ""
fi

# Step 3: Cross-Modal Decoding
echo "======================================================================"
echo "Step 3: Cross-Modal Analysis (Visual→Behavior) (~10 min)"
echo "======================================================================"
read -p "Run cross-modal decoding? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python experiments/cross_modal/visual_behavior_decoding.py \
        --sae-model "$SAE_MODEL" \
        --session-id $SESSION_ID \
        --allen-cache "$ALLEN_CACHE" \
        --output-dir results/cross_modal \
        --device "$DEVICE"

    echo ""
    echo "✓ Cross-modal analysis complete! Check results/cross_modal/"
    echo ""
fi

# Step 4: Feature Dynamics
echo "======================================================================"
echo "Step 4: Temporal Dynamics Analysis (~20 min)"
echo "======================================================================"
read -p "Run dynamics analysis? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python experiments/dynamics/feature_dynamics.py \
        --sae-model "$SAE_MODEL" \
        --session-id $SESSION_ID \
        --allen-cache "$ALLEN_CACHE" \
        --output-dir results/dynamics \
        --top-features 10 \
        --device "$DEVICE"

    echo ""
    echo "✓ Dynamics analysis complete! Check results/dynamics/"
    echo ""
fi

echo "======================================================================"
echo "Pipeline Complete!"
echo "======================================================================"
echo ""
echo "Results generated:"
echo "  - results/circuits/           (✅ done: attribution + motifs)"
echo "  - results/circuits/ablation/  (causal validation)"
echo "  - results/cross_modal/        (behavior decoding)"
echo "  - results/dynamics/           (temporal properties)"
echo ""
echo "Next steps:"
echo "  1. Review visualizations in results/*/*.png"
echo "  2. Read JSON results for detailed metrics"
echo "  3. Check CIRCUIT_EXTRACTION_GUIDE.md for interpretation"
echo "  4. Run on more sessions for multi-session validation"
echo ""
echo "🧠✨ Ready for publication!"
