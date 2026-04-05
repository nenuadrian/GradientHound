"""Export a PyTorch model to a rich JSON descriptor for visualization.

Combines module-tree metadata (from ``graph.py``) with a full ATen-level FX
computation graph captured via ``torch.export``.  The resulting JSON can be
loaded by the GradientHound Dash app to render interactive architecture
diagrams with per-op shapes, dtypes, and dataflow edges.

Usage::

    import torch, gradienthound

    model = MyModel()
    sample = (torch.randn(1, 3, 224, 224),)
    gradienthound.export_model(model, sample, "model.gh.json")
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

FORMAT_VERSION = "1.0"


# ── Helpers ────────────────────────────────────────────────────────────

def _tensor_meta(val: Any) -> dict | list | None:
    """Extract shape/dtype from a FakeTensor, real Tensor, or nested structure."""
    import torch

    if isinstance(val, torch.Tensor):
        return {
            "shape": list(val.shape),
            "dtype": str(val.dtype).removeprefix("torch."),
        }
    if isinstance(val, (tuple, list)):
        items = [_tensor_meta(v) for v in val]
        return items if any(i is not None for i in items) else None
    return None


def _node_target_str(node) -> str:
    """Stringify a node target to a readable op name."""
    target = node.target
    if callable(target):
        # aten ops: torch.ops.aten.conv2d.default → "aten.conv2d.default"
        name = getattr(target, "__name__", None)
        if name:
            overload = getattr(target, "overloadpacket", None)
            if overload:
                ns = getattr(overload, "__module__", "")
                # e.g. torch.ops.aten → "aten"
                ns = ns.replace("torch.ops.", "")
                return f"{ns}.{name}"
            module = getattr(target, "__module__", "") or ""
            qualname = getattr(target, "__qualname__", name) or name
            if module:
                return f"{module}.{qualname}"
            return qualname
        return repr(target)
    return str(target)


def _arg_names(args: tuple, kwargs: dict) -> list[str]:
    """Extract node name references from args, skipping constants."""
    import torch.fx

    names: list[str] = []
    for a in args:
        if isinstance(a, torch.fx.Node):
            names.append(a.name)
        elif isinstance(a, (tuple, list)):
            for item in a:
                if isinstance(item, torch.fx.Node):
                    names.append(item.name)
    for v in kwargs.values():
        if isinstance(v, torch.fx.Node):
            names.append(v.name)
    return names


def _nn_module_path(node) -> tuple[str | None, str | None]:
    """Extract the innermost nn.Module path and type from node metadata."""
    stack = node.meta.get("nn_module_stack")
    if not stack:
        return None, None
    # nn_module_stack is an OrderedDict: key → (qualname, type)
    # The last entry is the innermost module.
    last_key = list(stack.keys())[-1]
    qualname, mod_type = stack[last_key]
    type_str = mod_type if isinstance(mod_type, str) else (
        f"{mod_type.__module__}.{mod_type.__qualname__}"
        if hasattr(mod_type, "__qualname__") else str(mod_type)
    )
    return qualname, type_str


def _source_fn(node) -> str | None:
    """Extract the high-level source function name if available."""
    stack = node.meta.get("source_fn_stack")
    if not stack:
        return None
    # source_fn_stack: list of (name, target) tuples
    _name, target = stack[-1]
    if callable(target):
        return getattr(target, "__name__", str(target))
    return str(target)


# ── Parameter collection ──────────────────────────────────────────────

def _collect_parameters(model: nn.Module) -> dict[str, dict]:
    """Collect shape/dtype/device metadata for every parameter and buffer."""
    params: dict[str, dict] = {}
    for name, p in model.named_parameters():
        params[name] = {
            "shape": list(p.shape),
            "dtype": str(p.dtype).removeprefix("torch."),
            "device": str(p.device),
            "requires_grad": p.requires_grad,
            "numel": p.numel(),
        }
    for name, b in model.named_buffers():
        params[name] = {
            "shape": list(b.shape),
            "dtype": str(b.dtype).removeprefix("torch."),
            "device": str(b.device),
            "requires_grad": False,
            "numel": b.numel(),
            "is_buffer": True,
        }
    return params


# ── FX graph extraction ──────────────────────────────────────────────

def _extract_fx_graph(exported) -> dict:
    """Walk the exported FX graph and return nodes + edges."""
    nodes: list[dict] = []
    edges: list[dict] = []

    for node in exported.graph_module.graph.nodes:
        # --- per-node data ------------------------------------------------
        entry: dict[str, Any] = {
            "name": node.name,
            "op": node.op,
            "target": _node_target_str(node) if node.op != "output" else "output",
        }

        # Input references (for edges)
        input_names = _arg_names(node.args, node.kwargs)
        entry["args"] = input_names

        # Output tensor metadata
        val = node.meta.get("val")
        if val is not None:
            meta = _tensor_meta(val)
            if isinstance(meta, dict):
                entry["output_shape"] = meta["shape"]
                entry["output_dtype"] = meta["dtype"]
            elif isinstance(meta, list):
                # Multi-output node
                entry["outputs"] = [m for m in meta if m is not None]

        # Module mapping
        mod_path, mod_type = _nn_module_path(node)
        entry["nn_module"] = mod_path
        entry["nn_module_type"] = mod_type

        # Source function
        entry["source_fn"] = _source_fn(node)

        # Source location
        stack_trace = node.meta.get("stack_trace")
        if stack_trace:
            # Keep only the last frame (most relevant)
            lines = stack_trace.strip().splitlines()
            entry["stack_trace"] = lines[-1].strip() if lines else None

        nodes.append(entry)

        # --- edges --------------------------------------------------------
        for src in input_names:
            edges.append({"from": src, "to": node.name})

    return {"nodes": nodes, "edges": edges}


# ── Graph signature ──────────────────────────────────────────────────

def _extract_signature(exported) -> dict:
    """Extract the graph signature: params, buffers, user inputs/outputs."""
    sig = exported.graph_signature

    result: dict[str, Any] = {}

    # Input spec
    params = []
    buffers = []
    user_inputs = []
    for spec in sig.input_specs:
        kind = spec.kind.name  # PARAMETER, BUFFER, USER_INPUT, etc.
        target = spec.target if hasattr(spec, "target") else str(spec.arg)
        if kind == "PARAMETER":
            params.append(target)
        elif kind == "BUFFER":
            buffers.append(target)
        elif kind == "USER_INPUT":
            user_inputs.append(spec.arg.name if hasattr(spec.arg, "name") else str(spec.arg))

    # Output spec
    user_outputs = []
    for spec in sig.output_specs:
        kind = spec.kind.name
        if kind == "USER_OUTPUT":
            user_outputs.append(spec.arg.name if hasattr(spec.arg, "name") else str(spec.arg))

    result["parameters"] = params
    result["buffers"] = buffers
    result["user_inputs"] = user_inputs
    result["user_outputs"] = user_outputs

    return result


# ── Input / output specs from example tensors ────────────────────────

def _input_specs(example_inputs: tuple) -> list[dict]:
    import torch
    specs = []
    for i, inp in enumerate(example_inputs):
        if isinstance(inp, torch.Tensor):
            specs.append({
                "name": f"input_{i}",
                "shape": list(inp.shape),
                "dtype": str(inp.dtype).removeprefix("torch."),
            })
    return specs


# ── Public API ────────────────────────────────────────────────────────

def export_model(
    model: nn.Module,
    example_inputs: tuple,
    output: str | Path | None = None,
    *,
    dynamic_shapes: dict | None = None,
    strict: bool = True,
) -> dict:
    """Export a PyTorch model to a rich JSON descriptor.

    Args:
        model: The ``nn.Module`` to export.
        example_inputs: A tuple of example input tensors (same as you'd pass
            to ``model(*example_inputs)``).
        output: Optional file path.  When given, the descriptor is written as
            JSON to this path (conventionally ``*.gh.json``).
        dynamic_shapes: Passed through to ``torch.export.export()`` for
            symbolic shape constraints.
        strict: Passed through to ``torch.export.export()``.  Set ``False``
            for models with data-dependent control flow.

    Returns:
        The descriptor dictionary (same data written to *output*).
    """
    import torch

    from gradienthound.graph import extract_model_graph

    model_name = type(model).__name__
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"

    # ── Step 1: module tree (always works) ────────────────────────────
    module_tree = extract_model_graph(model_name, model)

    # ── Step 2: parameter metadata ────────────────────────────────────
    parameters = _collect_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Step 3: FX graph via torch.export ─────────────────────────────
    fx_graph = None
    graph_signature = None

    try:
        exported = torch.export.export(
            model,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        )
        fx_graph = _extract_fx_graph(exported)
        graph_signature = _extract_signature(exported)
    except Exception as exc:
        warnings.warn(
            f"torch.export failed ({exc!r}); falling back to module-tree only. "
            f"The FX computation graph will not be available.",
            stacklevel=2,
        )

    # ── Step 4: assemble descriptor ───────────────────────────────────
    descriptor: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_name": model_name,
        "model_class": model_class,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "inputs": _input_specs(example_inputs),
        "outputs": [],
        "module_tree": module_tree,
        "parameters": parameters,
    }

    if fx_graph is not None:
        descriptor["fx_graph"] = fx_graph
    if graph_signature is not None:
        descriptor["graph_signature"] = graph_signature
        # Fill in output specs from the FX graph output nodes
        for node in (fx_graph or {}).get("nodes", []):
            if node["op"] == "output" and node.get("outputs"):
                descriptor["outputs"] = node["outputs"]
            elif node["op"] == "output" and node.get("output_shape"):
                descriptor["outputs"] = [{
                    "shape": node["output_shape"],
                    "dtype": node.get("output_dtype", "unknown"),
                }]

    # Also try to get outputs from user_output nodes
    if not descriptor["outputs"] and fx_graph:
        for node in fx_graph["nodes"]:
            if node["name"] in (graph_signature or {}).get("user_outputs", []):
                if node.get("output_shape"):
                    descriptor["outputs"].append({
                        "name": node["name"],
                        "shape": node["output_shape"],
                        "dtype": node.get("output_dtype", "unknown"),
                    })

    # ── Step 5: serialize ─────────────────────────────────────────────
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(descriptor, f, indent=2, default=str)

    return descriptor
