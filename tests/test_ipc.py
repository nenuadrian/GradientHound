"""Tests for the SQLite-backed IPCChannel."""
from __future__ import annotations

import json
import threading
import time

import pytest

from gradienthound.ipc import IPCChannel


@pytest.fixture()
def ipc(tmp_path):
    """Create a fresh IPCChannel in a temp directory."""
    ch = IPCChannel(directory=tmp_path / "ipc_test")
    yield ch
    ch.cleanup()


# ── Key-value store (metadata, models, optimizers) ───────────────────


class TestKVStore:
    def test_write_and_read_metadata(self, ipc):
        ipc.write_metadata({"project": "test", "run_id": "abc"})
        assert ipc.read_metadata() == {"project": "test", "run_id": "abc"}

    def test_read_missing_returns_empty(self, ipc):
        assert ipc.read_metadata() == {}
        assert ipc.read_models() == {}
        assert ipc.read_optimizers() == {}

    def test_overwrite_metadata(self, ipc):
        ipc.write_metadata({"v": 1})
        ipc.write_metadata({"v": 2})
        assert ipc.read_metadata() == {"v": 2}

    def test_models_round_trip(self, ipc):
        models = {"net": {"type": "CNN", "parameters": {"conv1.weight": {"shape": [8, 3, 3, 3]}}}}
        ipc.write_models(models)
        assert ipc.read_models() == models

    def test_optimizers_round_trip(self, ipc):
        opts = {"adam": {"type": "Adam", "defaults": {"lr": 0.001}}}
        ipc.write_optimizers(opts)
        assert ipc.read_optimizers() == opts


# ── Event append / read ──────────────────────────────────────────────


class TestGradientStats:
    def test_append_and_read(self, ipc):
        entries = [
            {"step": 1, "model": "net", "layer": "conv1.weight", "grad_norm": 0.5, "_timestamp": 1.0},
            {"step": 1, "model": "net", "layer": "fc.weight", "grad_norm": 0.3, "_timestamp": 1.0},
        ]
        ipc.append_gradient_stats(entries)
        result = ipc.read_gradient_stats()
        assert len(result) == 2
        assert result[0]["layer"] == "conv1.weight"
        assert result[1]["layer"] == "fc.weight"

    def test_read_empty(self, ipc):
        assert ipc.read_gradient_stats() == []

    def test_filter_by_model(self, ipc):
        ipc.append_gradient_stats([
            {"step": 1, "model": "A", "layer": "x", "grad_norm": 1.0},
            {"step": 1, "model": "B", "layer": "x", "grad_norm": 2.0},
            {"step": 2, "model": "A", "layer": "x", "grad_norm": 3.0},
        ])
        result = ipc.read_gradient_stats(model="A")
        assert len(result) == 2
        assert all(e["model"] == "A" for e in result)

    def test_filter_by_step_range(self, ipc):
        for s in range(1, 11):
            ipc.append_gradient_stats([{"step": s, "model": "net", "layer": "w", "grad_norm": float(s)}])

        result = ipc.read_gradient_stats(step_min=5, step_max=8)
        assert len(result) == 4
        steps = [e["step"] for e in result]
        assert steps == [5, 6, 7, 8]

    def test_filter_by_layer(self, ipc):
        ipc.append_gradient_stats([
            {"step": 1, "model": "net", "layer": "conv1.weight", "grad_norm": 1.0},
            {"step": 1, "model": "net", "layer": "fc.weight", "grad_norm": 2.0},
        ])
        result = ipc.read_gradient_stats(layer="fc.weight")
        assert len(result) == 1
        assert result[0]["layer"] == "fc.weight"

    def test_last_n(self, ipc):
        for s in range(1, 21):
            ipc.append_gradient_stats([{"step": s, "model": "net", "layer": "w"}])

        result = ipc.read_gradient_stats(last_n=5)
        assert len(result) == 5
        # Should be the LAST 5, in ascending order.
        assert [e["step"] for e in result] == [16, 17, 18, 19, 20]

    def test_combined_filters(self, ipc):
        for s in range(1, 11):
            ipc.append_gradient_stats([
                {"step": s, "model": "A", "layer": "w"},
                {"step": s, "model": "B", "layer": "w"},
            ])
        result = ipc.read_gradient_stats(model="A", step_min=3, step_max=5)
        assert len(result) == 3
        assert all(e["model"] == "A" for e in result)


class TestMetrics:
    def test_append_and_read(self, ipc):
        ipc.append_metrics({"_step": 1, "loss": 0.5, "_timestamp": 1.0})
        ipc.append_metrics({"_step": 2, "loss": 0.3, "_timestamp": 2.0})
        result = ipc.read_metrics()
        assert len(result) == 2
        assert result[0]["loss"] == 0.5
        assert result[1]["loss"] == 0.3

    def test_step_filter_uses_underscore_step(self, ipc):
        """Metrics use ``_step`` instead of ``step``.  The step column
        should be populated from ``_step`` so SQL filters work."""
        ipc.append_metrics({"_step": 10, "loss": 1.0})
        ipc.append_metrics({"_step": 20, "loss": 2.0})
        ipc.append_metrics({"_step": 30, "loss": 3.0})
        result = ipc.read_metrics(step_min=15)
        assert len(result) == 2
        assert result[0]["_step"] == 20


class TestWeightStats:
    def test_round_trip(self, ipc):
        entries = [{"step": 1, "model": "net", "layer": "w", "norm_l2": 5.0, "_timestamp": 1.0}]
        ipc.append_weight_stats(entries)
        result = ipc.read_weight_stats()
        assert len(result) == 1
        assert result[0]["norm_l2"] == 5.0


class TestActivationStats:
    def test_round_trip(self, ipc):
        entries = [{"step": 1, "model": "net", "layer": "relu", "mean": 0.5, "_timestamp": 1.0}]
        ipc.append_activation_stats(entries)
        assert len(ipc.read_activation_stats()) == 1


class TestPredictions:
    def test_round_trip(self, ipc):
        ipc.append_predictions([{"step": 1, "name": "val", "predicted": [1.0], "actual": [1.1]}])
        result = ipc.read_predictions()
        assert len(result) == 1
        assert result[0]["predicted"] == [1.0]


class TestAttention:
    def test_round_trip(self, ipc):
        ipc.append_attention([{"step": 1, "name": "attn", "heads": 4, "weights": [[0.1]]}])
        result = ipc.read_attention()
        assert len(result) == 1
        assert result[0]["heads"] == 4


class TestOptimizerState:
    def test_round_trip(self, ipc):
        ipc.append_optimizer_state([{
            "step": 1, "optimizer": "adam", "type": "Adam",
            "groups": [{"lr": 0.001}], "_timestamp": 1.0,
        }])
        result = ipc.read_optimizer_state()
        assert len(result) == 1
        assert result[0]["type"] == "Adam"


# ── Utility methods ──────────────────────────────────────────────────


class TestUtilities:
    def test_max_step(self, ipc):
        assert ipc._max_step("gradient_stats") is None

        ipc.append_gradient_stats([
            {"step": 5, "model": "A", "layer": "w"},
            {"step": 10, "model": "A", "layer": "w"},
            {"step": 8, "model": "B", "layer": "w"},
        ])
        assert ipc._max_step("gradient_stats") == 10
        assert ipc._max_step("gradient_stats", model="A") == 10
        assert ipc._max_step("gradient_stats", model="B") == 8
        assert ipc._max_step("gradient_stats", model="C") is None

    def test_count_events(self, ipc):
        assert ipc._count_events("gradient_stats") == 0
        ipc.append_gradient_stats([{"step": 1}, {"step": 2}])
        assert ipc._count_events("gradient_stats") == 2
        assert ipc._count_events("weight_stats") == 0


# ── Request / response ───────────────────────────────────────────────


class TestRequestResponse:
    def test_request_lifecycle(self, ipc):
        assert ipc.read_requests() == []

        ipc.write_request({"type": "weight_heatmap", "id": "req1", "model": "net", "layer": "w"})
        ipc.write_request({"type": "cka", "id": "req2", "model": "net"})

        reqs = ipc.read_requests()
        assert len(reqs) == 2
        assert reqs[0]["id"] == "req1"
        assert reqs[1]["id"] == "req2"

        ipc.clear_requests()
        assert ipc.read_requests() == []

    def test_response_lifecycle(self, ipc):
        assert ipc.read_response("missing") is None

        ipc.write_response("req1", {"step": 1, "matrix": [[1, 2], [3, 4]]})
        resp = ipc.read_response("req1")
        assert resp["matrix"] == [[1, 2], [3, 4]]

        ipc.clear_response("req1")
        assert ipc.read_response("req1") is None


# ── Concurrent access ────────────────────────────────────────────────


class TestConcurrency:
    def test_concurrent_write_and_read(self, tmp_path):
        """Simulate training writes + dashboard reads in separate threads."""
        directory = tmp_path / "concurrent"
        writer = IPCChannel(directory=directory)
        reader = IPCChannel(directory=directory)

        results = {"written": 0, "read": 0}
        errors = []

        def write_worker():
            try:
                for s in range(100):
                    writer.append_gradient_stats([
                        {"step": s, "model": "net", "layer": "w", "grad_norm": float(s)},
                    ])
                    results["written"] += 1
            except Exception as exc:
                errors.append(exc)

        def read_worker():
            try:
                for _ in range(50):
                    entries = reader.read_gradient_stats()
                    results["read"] = max(results["read"], len(entries))
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        t_write = threading.Thread(target=write_worker)
        t_read = threading.Thread(target=read_worker)
        t_write.start()
        t_read.start()
        t_write.join()
        t_read.join()

        writer.cleanup()

        assert not errors, f"Concurrency errors: {errors}"
        assert results["written"] == 100
        # Reader should have seen at least some entries.
        assert results["read"] > 0


# ── Cleanup ──────────────────────────────────────────────────────────


class TestCleanup:
    def test_cleanup_temp_dir(self):
        ch = IPCChannel()
        d = ch.directory
        assert d.exists()
        ch.cleanup()
        assert not d.exists()

    def test_cleanup_does_not_remove_user_dir(self, tmp_path):
        d = tmp_path / "mydata"
        ch = IPCChannel(directory=d)
        ch.write_metadata({"x": 1})
        ch.cleanup()
        # Directory created by us but not owned -- should NOT be removed.
        assert d.exists()

    def test_db_file_exists(self, ipc):
        db = ipc.directory / "gradienthound.db"
        assert db.exists()


# ── Insertion order ──────────────────────────────────────────────────


class TestInsertionOrder:
    def test_events_returned_in_insertion_order(self, ipc):
        """Events appended across multiple calls should come back in order."""
        ipc.append_gradient_stats([{"step": 3, "model": "net", "layer": "a"}])
        ipc.append_gradient_stats([{"step": 1, "model": "net", "layer": "b"}])
        ipc.append_gradient_stats([{"step": 2, "model": "net", "layer": "c"}])

        result = ipc.read_gradient_stats()
        assert [e["layer"] for e in result] == ["a", "b", "c"]
