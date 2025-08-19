from neuros.db.database import Database


def test_database_insert_and_retrieve():
    db = Database(db_path=":memory:")
    run_id = "test_run"
    metrics = {
        "duration": 1.0,
        "samples": 10,
        "throughput": 10.0,
        "mean_latency": 0.001,
        "accuracy": 0.5,
    }
    db.insert_run_metrics(run_id, metrics)
    retrieved = db.get_run_metrics(run_id)
    assert retrieved is not None
    assert retrieved["samples"] == 10
    # insert results
    results = [(0.0, 1, 0.8, 0.001), (0.1, 0, 0.6, 0.002)]
    db.insert_stream_results(run_id, results)
    res = db.get_stream_results(run_id)
    assert len(res) == 2