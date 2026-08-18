from pathlib import Path

import numpy as np
import yaml

from orion.tokenization import (
    AssemblyTokenizer,
    BinnedCountTokenizer,
    BurstTokenizer,
    EventSpikeTokenizer,
    ISIRelativeTimeTokenizer,
    SynchronyPacketTokenizer,
    VQMotifTokenizer,
    generate_synthetic_session,
    run_synthetic_benchmark,
    write_benchmark_reports,
)


def test_synthetic_generator_is_deterministic_and_contains_all_motifs():
    first = generate_synthetic_session(seed=7)
    second = generate_synthetic_session(seed=7)
    assert first.events == second.events
    assert first.motifs == second.motifs
    assert {motif.label for motif in first.motifs} == {
        "burst",
        "synchrony",
        "assembly",
        "leader_chain",
        "pause_rebound",
        "movement_volley",
    }


def test_interpretable_tokenizers_preserve_expected_semantics():
    session = generate_synthetic_session(seed=3)
    event = EventSpikeTokenizer().encode_events(session.events)
    binned = BinnedCountTokenizer(bin_ms=10).encode_events(session.events)
    isi = ISIRelativeTimeTokenizer().encode_events(session.events)
    burst = BurstTokenizer().encode_events(session.events)
    synchrony = SynchronyPacketTokenizer().encode_events(session.events)

    assert len(event.token_ids) == len(session.events)
    assert len(binned.token_ids) < len(event.token_ids)
    assert np.any(isi.side_features["kind"] == 0)  # WAIT tokens
    assert np.any(burst.side_features["kind"] == 1)  # BURST tokens
    assert np.any(burst.side_features["kind"] == 2)  # PAUSE tokens
    assert np.any(synchrony.side_features["kind"] == 1)  # population packets


def test_fit_requiring_tokenizers_are_deterministic_and_noncollapsed():
    train = generate_synthetic_session(seed=11)
    test = generate_synthetic_session(seed=12)

    vq_a = VQMotifTokenizer(codebook_size=8, seed=5).fit_events(train.events)
    vq_b = VQMotifTokenizer(codebook_size=8, seed=5).fit_events(train.events)
    np.testing.assert_allclose(vq_a.codebook_, vq_b.codebook_)
    vq_batch = vq_a.encode_events(test.events)
    assert vq_batch.metadata["active_codes"] > 1

    assembly_a = AssemblyTokenizer(n_assemblies=3).fit_events(train.events)
    assembly_b = AssemblyTokenizer(n_assemblies=3).fit_events(train.events)
    np.testing.assert_allclose(assembly_a.components_, assembly_b.components_)
    assert assembly_a.summaries_ == assembly_b.summaries_
    assert len(assembly_a.summaries_) == 3
    assembly_batch = assembly_a.encode_events(test.events)
    assert len(assembly_batch.token_ids) > 0


def test_synthetic_benchmark_scores_all_tokenizers_without_leakage(tmp_path: Path):
    config = yaml.safe_load(Path("configs/orion/tokenization_smoke.yaml").read_text())
    report = run_synthetic_benchmark(config)
    scores = report["scores"]
    assert {score["tokenizer"] for score in scores} == {
        "event",
        "binned_count",
        "isi_relative",
        "burst",
        "synchrony",
        "vq_motif",
        "assembly",
    }
    for score in scores:
        assert score["input_events"] > 0
        assert score["token_count"] > 0
        assert 0.0 <= score["motif_decoding_accuracy"] <= 1.0
        assert 0.0 <= score["jitter_similarity"] <= 1.0
        assert 0.0 <= score["unit_dropout_similarity"] <= 1.0
        assert score["encode_ms"] >= 0.0

    write_benchmark_reports(report, tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "comparison_table.csv").exists()
    cards = (tmp_path / "tokenizer_cards.md").read_text()
    assert "## event" in cards
    assert "## vq_motif" in cards
