from __future__ import annotations

from gradienthound.checkpoint import annotate_checkpoint_events


def test_annotate_checkpoint_events_detects_rank_kurtosis_and_norm_outlier() -> None:
    snapshots = [
        {
            "name": "ckpt_0",
            "weight_stats": [
                {"layer": "layer.a", "norm_l2": 10.0, "effective_rank": 12.0, "kurtosis": 0.1},
                {"layer": "layer.b", "norm_l2": 10.0, "effective_rank": 10.0, "kurtosis": 0.0},
                {"layer": "layer.c", "norm_l2": 10.0, "effective_rank": 10.0, "kurtosis": 0.0},
                {"layer": "layer.d", "norm_l2": 10.0, "effective_rank": 10.0, "kurtosis": 0.0},
                {"layer": "layer.e", "norm_l2": 10.0, "effective_rank": 10.0, "kurtosis": 0.0},
                {"layer": "layer.f", "norm_l2": 10.0, "effective_rank": 10.0, "kurtosis": 0.0},
            ],
        },
        {
            "name": "ckpt_1",
            "weight_stats": [
                {"layer": "layer.a", "norm_l2": 10.2, "effective_rank": 4.0, "kurtosis": 4.5},
                {"layer": "layer.b", "norm_l2": 10.1, "effective_rank": 10.0, "kurtosis": 0.1},
                {"layer": "layer.c", "norm_l2": 10.1, "effective_rank": 10.0, "kurtosis": 0.0},
                {"layer": "layer.d", "norm_l2": 10.1, "effective_rank": 10.0, "kurtosis": 0.2},
                {"layer": "layer.e", "norm_l2": 10.1, "effective_rank": 10.0, "kurtosis": 0.1},
                {"layer": "layer.f", "norm_l2": 20.0, "effective_rank": 10.0, "kurtosis": 0.0},
            ],
        },
    ]

    annotate_checkpoint_events(snapshots)

    anomalies = snapshots[1].get("anomalies", [])
    event_types = {a["type"] for a in anomalies}

    assert "rank_collapse" in event_types
    assert "kurtosis_spike" in event_types
    assert "norm_jump_outlier" in event_types

    # Ensure ranking is descending by score.
    scores = [a["score"] for a in anomalies]
    assert scores == sorted(scores, reverse=True)


def test_annotate_checkpoint_events_is_noop_for_single_snapshot() -> None:
    snapshots = [
        {
            "name": "ckpt_0",
            "weight_stats": [
                {"layer": "layer.a", "norm_l2": 1.0, "effective_rank": 1.0, "kurtosis": 0.0},
            ],
        }
    ]

    annotate_checkpoint_events(snapshots)

    assert "anomalies" not in snapshots[0]
    assert "anomaly_summary" not in snapshots[0]