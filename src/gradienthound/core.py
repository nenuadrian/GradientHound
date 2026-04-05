from __future__ import annotations

import atexit
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .graph import extract_model_graph
from .hooks import WatchState
from .ipc import IPCChannel

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim


class GradientHound:
    """Main entry point for GradientHound.

    Args:
        ui: If True, launch a Streamlit dashboard in the background.
        port: Specific port for Streamlit. If None, picks a free port.
    """

    def __init__(
        self,
        ui: bool = False,
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._models: dict[str, dict] = {}
        self._optimizers: dict[str, dict] = {}
        self._optimizer_refs: dict[str, optim.Optimizer] = {}
        self._watches: dict[str, WatchState] = {}
        self._metadata: dict = metadata or {}
        self._ipc = IPCChannel()
        self._process: subprocess.Popen | None = None
        self._ui = ui
        self._wandb_original_log: Any = None
        self._step: int = 0

        self._ipc.write_metadata(self._metadata)

        if ui:
            self._start_ui(port)

        atexit.register(self.shutdown)

    def _start_ui(self, port: int | None = None) -> None:
        if port is None:
            port = _find_free_port()

        app_path = Path(__file__).parent / "_streamlit_app.py"

        self._process = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run",
                str(app_path),
                "--server.headless", "true",
                "--server.port", str(port),
                "--server.fileWatcherType", "none",
                "--browser.gatherUsageStats", "false",
                "--",
                "--ipc-dir", str(self._ipc.directory),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        url = f"http://localhost:{port}"
        print(f"GradientHound UI: {url}")
        webbrowser.open_new_tab(url)

    def register_model(self, name: str, model: nn.Module) -> None:
        """Register a PyTorch model. The UI will display its architecture."""
        graph = extract_model_graph(name, model)
        self._models[name] = graph
        self._ipc.write_models(self._models)

    def register_optimizer(self, name: str, optimizer: optim.Optimizer) -> None:
        """Register an optimizer. The UI will display its configuration."""
        info = _extract_optimizer_info(optimizer)
        self._optimizers[name] = info
        self._optimizer_refs[name] = optimizer
        self._ipc.write_optimizers(self._optimizers)

    # ── Hook-based capture ──────────────────────────────────────────

    def watch(
        self,
        model: nn.Module,
        name: str | None = None,
        *,
        log_gradients: bool = True,
        log_activations: bool = False,
        weight_every: int = 50,
    ) -> None:
        """Register PyTorch hooks for automatic gradient/weight/activation capture."""
        if name is None:
            name = type(model).__name__
        if name not in self._models:
            self.register_model(name, model)
        self._watches[name] = WatchState(
            model, name,
            log_gradients=log_gradients,
            log_activations=log_activations,
            weight_every=weight_every,
        )

    def step(self) -> None:
        """Called each training step.  Flushes buffered stats to IPC."""
        self._step += 1
        for ws in self._watches.values():
            ws.flush_gradient_stats(self._step, self._ipc)
            ws.flush_activation_stats(self._step, self._ipc)
            if self._step % ws.weight_every == 0:
                ws.compute_weight_stats(self._step, self._ipc)
                self._flush_optimizer_state()
            ws.process_requests(self._step, self._ipc)

    def _flush_optimizer_state(self) -> None:
        """Extract live optimizer state statistics and write to IPC."""
        entries = []
        for name, optimizer in self._optimizer_refs.items():
            entry = _extract_optimizer_state_stats(optimizer, self._step, name)
            if entry:
                entries.append(entry)
        if entries:
            self._ipc.append_optimizer_state(entries)

    def log_weights(self, name: str | None = None) -> None:
        """Force an immediate weight snapshot."""
        targets = [self._watches[name]] if name else list(self._watches.values())
        for ws in targets:
            ws.compute_weight_stats(self._step, self._ipc)

    def log_attention(
        self,
        name: str,
        weights: torch.Tensor,
    ) -> None:
        """Log attention weight matrix for visualization.

        *weights* shape: ``(batch, heads, seq_q, seq_kv)`` or ``(heads, seq_q, seq_kv)``.
        Large sequences are downsampled to 64x64.
        """
        import torch
        import torch.nn.functional as F

        w = weights.detach().float()
        if w.ndim == 4:
            w = w[0]  # take first batch element
        # w is now (heads, seq_q, seq_kv)
        heads, sq, skv = w.shape

        # Downsample large sequences
        max_seq = 64
        if sq > max_seq or skv > max_seq:
            w = F.adaptive_avg_pool2d(w.unsqueeze(0), (min(sq, max_seq), min(skv, max_seq))).squeeze(0)

        self._ipc.append_attention([{
            "step": self._step,
            "name": name,
            "heads": heads,
            "shape": [heads, sq, skv],
            "weights": w.tolist(),
            "_timestamp": time.time(),
        }])

    def log_predictions(
        self,
        predicted: Any,
        actual: Any,
        name: str = "default",
    ) -> None:
        """Log prediction vs actual for calibration scatter plot."""
        import torch

        def _to_list(x: Any) -> list[float]:
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().flatten().tolist()
            if isinstance(x, (list, tuple)):
                return [float(v) for v in x]
            return [float(x)]

        pred = _to_list(predicted)
        act = _to_list(actual)
        # Subsample if too many points
        if len(pred) > 500:
            stride = len(pred) // 500
            pred = pred[::stride]
            act = act[::stride]

        self._ipc.append_predictions([{
            "step": self._step,
            "name": name,
            "predicted": pred,
            "actual": act,
            "_timestamp": time.time(),
        }])

    # ── Wandb capture ─────────────────────────────────────────────────

    def capture_wandb(self) -> None:
        """Monkey-patch wandb.log to also capture metrics for the GradientHound UI.

        Call this after wandb.init(). All subsequent wandb.log() calls will
        also be recorded and displayed as live time-series charts.
        """
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")

        if self._wandb_original_log is not None:
            return  # already patched

        original_log = wandb.log
        self._wandb_original_log = original_log

        def _patched_log(data: dict, *args: Any, **kwargs: Any) -> Any:
            # Forward to real wandb.log first
            result = original_log(data, *args, **kwargs)

            # Extract scalar values for our dashboard
            entry: dict[str, Any] = {"_step": self._step, "_timestamp": time.time()}
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    entry[k] = v
            self._step += 1

            if len(entry) > 2:  # has more than just _step and _timestamp
                self._ipc.append_metrics(entry)

            return result

        wandb.log = _patched_log

    def _restore_wandb(self) -> None:
        """Restore the original wandb.log if we patched it."""
        if self._wandb_original_log is not None:
            try:
                import wandb
                wandb.log = self._wandb_original_log
            except ImportError:
                pass
            self._wandb_original_log = None

    def shutdown(self) -> None:
        """Stop the Streamlit subprocess and clean up."""
        self._restore_wandb()

        for ws in self._watches.values():
            ws.remove_hooks()
        self._watches.clear()

        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
            self._process = None

        self._ipc.cleanup()

    def __enter__(self) -> GradientHound:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


def _extract_optimizer_info(optimizer: optim.Optimizer) -> dict:
    defaults = {}
    for k, v in optimizer.defaults.items():
        if isinstance(v, (int, float, bool, str)):
            defaults[k] = v

    opt_type = type(optimizer).__name__
    # Determine state buffer multiplier for memory estimation
    adam_types = {"Adam", "AdamW", "NAdam", "RAdam", "Adagrad", "Adadelta"}
    if opt_type in adam_types:
        buffers_per_param = 2  # exp_avg + exp_avg_sq
    elif opt_type in {"SGD", "ASGD"} and defaults.get("momentum", 0) > 0:
        buffers_per_param = 1  # momentum_buffer
    elif opt_type == "RMSprop":
        buffers_per_param = 2 if defaults.get("momentum", 0) > 0 else 1
    else:
        buffers_per_param = 0

    total_numel = 0
    param_groups = []
    for i, group in enumerate(optimizer.param_groups):
        group_numel = sum(p.numel() for p in group["params"])
        # Estimate bytes: param element size * numel * buffers
        elem_size = 4  # default float32
        if group["params"]:
            elem_size = group["params"][0].element_size()
        group_memory_bytes = group_numel * elem_size * buffers_per_param

        pg: dict = {
            "index": i,
            "num_params": len(group["params"]),
            "total_numel": group_numel,
            "memory_bytes": group_memory_bytes,
        }
        for k, v in group.items():
            if k == "params":
                continue
            if isinstance(v, (int, float, bool, str)):
                pg[k] = v
        param_groups.append(pg)
        total_numel += group_numel

    # Back-fill percentage of total
    for pg in param_groups:
        pg["pct_of_total"] = round(pg["total_numel"] / max(total_numel, 1) * 100, 1)

    total_memory = sum(pg["memory_bytes"] for pg in param_groups)

    return {
        "type": opt_type,
        "defaults": defaults,
        "param_groups": param_groups,
        "total_numel": total_numel,
        "total_memory_bytes": total_memory,
        "buffers_per_param": buffers_per_param,
    }


def _extract_optimizer_state_stats(
    optimizer: optim.Optimizer, step: int, name: str,
) -> dict | None:
    """Extract per-group statistics from the live optimizer state buffers."""
    import time as _time

    state = optimizer.state
    if not state:
        return None

    opt_type = type(optimizer).__name__
    groups_stats = []

    for i, group in enumerate(optimizer.param_groups):
        exp_avg_norms: list[float] = []
        exp_avg_sq_means: list[float] = []
        momentum_norms: list[float] = []
        adam_step: int = 0

        for p in group["params"]:
            if p not in state:
                continue
            s = state[p]

            # Adam family
            if "exp_avg" in s:
                exp_avg_norms.append(s["exp_avg"].norm(2).item())
            if "exp_avg_sq" in s:
                exp_avg_sq_means.append(s["exp_avg_sq"].mean().item())
            if "step" in s:
                sv = s["step"]
                adam_step = max(adam_step, int(sv.item()) if hasattr(sv, "item") else int(sv))

            # SGD / RMSprop momentum
            if "momentum_buffer" in s:
                momentum_norms.append(s["momentum_buffer"].norm(2).item())

        gs: dict = {"group_index": i, "lr": group.get("lr", 0)}

        if exp_avg_norms:
            gs["exp_avg_norm_mean"] = sum(exp_avg_norms) / len(exp_avg_norms)
            gs["exp_avg_norm_max"] = max(exp_avg_norms)
        if exp_avg_sq_means:
            gs["exp_avg_sq_mean"] = sum(exp_avg_sq_means) / len(exp_avg_sq_means)
            # Effective LR estimate: lr / (sqrt(avg second moment) + eps)
            eps = group.get("eps", 1e-8)
            avg_v = sum(exp_avg_sq_means) / len(exp_avg_sq_means)
            gs["effective_lr"] = group.get("lr", 0) / (avg_v ** 0.5 + eps)
        if momentum_norms:
            gs["momentum_norm_mean"] = sum(momentum_norms) / len(momentum_norms)
            gs["momentum_norm_max"] = max(momentum_norms)
        if adam_step > 0:
            gs["optimizer_step"] = adam_step
            # Warmup progress: bias correction factor converges as step grows
            betas = group.get("betas", (0.9, 0.999))
            if isinstance(betas, (list, tuple)) and len(betas) == 2:
                beta2 = betas[1]
                # Bias correction factor for variance: 1 - beta2^step
                bias_correction = 1 - beta2 ** adam_step
                gs["bias_correction2"] = bias_correction
                # ~99% corrected at step = log(0.01)/log(beta2)
                import math
                steps_to_converge = math.log(0.01) / math.log(beta2) if beta2 < 1 else 1000
                gs["warmup_pct"] = min(100.0, round(adam_step / steps_to_converge * 100, 1))

        groups_stats.append(gs)

    return {
        "step": step,
        "optimizer": name,
        "type": opt_type,
        "groups": groups_stats,
        "_timestamp": _time.time(),
    }


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
