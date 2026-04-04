from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path


class IPCChannel:
    """File-based IPC between the main process and the Streamlit subprocess."""

    def __init__(self, directory: Path | str | None = None) -> None:
        if directory is None:
            self._dir = Path(tempfile.mkdtemp(prefix="gradienthound_"))
            self._owns_dir = True
        else:
            self._dir = Path(directory)
            self._owns_dir = False

    @property
    def directory(self) -> Path:
        return self._dir

    # ── Atomic JSON read/write ────────────────────────────────────────

    def _atomic_write(self, filename: str, data: dict) -> None:
        target = self._dir / filename
        tmp = self._dir / f".{filename}.tmp"
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, target)

    def _atomic_read(self, filename: str) -> dict:
        target = self._dir / filename
        if not target.exists():
            return {}
        try:
            return json.loads(target.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    # ── Generic JSONL helpers ─────────────────────────────────────────

    def _append_jsonl(self, filename: str, entries: list[dict]) -> None:
        target = self._dir / filename
        with open(target, "a") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def _read_jsonl(self, filename: str) -> list[dict]:
        target = self._dir / filename
        if not target.exists():
            return []
        entries: list[dict] = []
        try:
            for line in target.read_text().splitlines():
                if line.strip():
                    entries.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass
        return entries

    # ── Metadata ──────────────────────────────────────────────────────

    def write_metadata(self, metadata: dict) -> None:
        self._atomic_write("metadata.json", metadata)

    def read_metadata(self) -> dict:
        return self._atomic_read("metadata.json")

    # ── Models ────────────────────────────────────────────────────────

    def write_models(self, models: dict[str, dict]) -> None:
        self._atomic_write("models.json", models)

    def read_models(self) -> dict[str, dict]:
        return self._atomic_read("models.json")

    # ── Optimizers ────────────────────────────────────────────────────

    def write_optimizers(self, optimizers: dict[str, dict]) -> None:
        self._atomic_write("optimizers.json", optimizers)

    def read_optimizers(self) -> dict[str, dict]:
        return self._atomic_read("optimizers.json")

    # ── Metrics (wandb scalars) ───────────────────────────────────────

    def append_metrics(self, entry: dict) -> None:
        self._append_jsonl("metrics.jsonl", [entry])

    def read_metrics(self) -> list[dict]:
        return self._read_jsonl("metrics.jsonl")

    # ── Gradient stats ────────────────────────────────────────────────

    def append_gradient_stats(self, entries: list[dict]) -> None:
        self._append_jsonl("gradient_stats.jsonl", entries)

    def read_gradient_stats(self) -> list[dict]:
        return self._read_jsonl("gradient_stats.jsonl")

    # ── Weight stats ──────────────────────────────────────────────────

    def append_weight_stats(self, entries: list[dict]) -> None:
        self._append_jsonl("weight_stats.jsonl", entries)

    def read_weight_stats(self) -> list[dict]:
        return self._read_jsonl("weight_stats.jsonl")

    # ── Activation stats ──────────────────────────────────────────────

    def append_activation_stats(self, entries: list[dict]) -> None:
        self._append_jsonl("activation_stats.jsonl", entries)

    def read_activation_stats(self) -> list[dict]:
        return self._read_jsonl("activation_stats.jsonl")

    # ── Predictions (value calibration) ───────────────────────────────

    def append_predictions(self, entries: list[dict]) -> None:
        self._append_jsonl("predictions.jsonl", entries)

    def read_predictions(self) -> list[dict]:
        return self._read_jsonl("predictions.jsonl")

    # ── Attention patterns ────────────────────────────────────────────

    def append_attention(self, entries: list[dict]) -> None:
        self._append_jsonl("attention.jsonl", entries)

    def read_attention(self) -> list[dict]:
        return self._read_jsonl("attention.jsonl")

    # ── On-demand requests / responses ───────────────────────────────

    def write_request(self, request: dict) -> None:
        """Queue a computation request for the training process."""
        self._append_jsonl("_requests.jsonl", [request])

    def read_requests(self) -> list[dict]:
        """Read all pending requests."""
        return self._read_jsonl("_requests.jsonl")

    def clear_requests(self) -> None:
        """Remove the request file (called by training process after processing)."""
        target = self._dir / "_requests.jsonl"
        if target.exists():
            target.unlink()

    def write_response(self, request_id: str, data: dict) -> None:
        """Write a computation response."""
        resp_dir = self._dir / "_responses"
        resp_dir.mkdir(exist_ok=True)
        target = resp_dir / f"{request_id}.json"
        tmp = resp_dir / f".{request_id}.json.tmp"
        tmp.write_text(json.dumps(data))
        os.replace(tmp, target)

    def read_response(self, request_id: str) -> dict | None:
        """Read a response if available.  Returns None if not yet computed."""
        target = self._dir / "_responses" / f"{request_id}.json"
        if not target.exists():
            return None
        try:
            return json.loads(target.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def clear_response(self, request_id: str) -> None:
        target = self._dir / "_responses" / f"{request_id}.json"
        if target.exists():
            target.unlink()

    # ── Cleanup ───────────────────────────────────────────────────────

    def cleanup(self) -> None:
        if self._owns_dir and self._dir.exists():
            shutil.rmtree(self._dir, ignore_errors=True)
