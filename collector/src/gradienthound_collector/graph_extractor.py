from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.fx

from .types import GraphNode, GraphEdge, ModelGraph

logger = logging.getLogger(__name__)


def _get_module_attributes(module: nn.Module) -> dict[str, Any]:
    """Extract constructor-relevant attributes from a module."""
    attrs = {}
    # Common attributes across module types
    for attr_name in [
        "in_features", "out_features",
        "in_channels", "out_channels", "kernel_size", "stride", "padding",
        "dilation", "groups", "bias",
        "num_features", "eps", "momentum", "affine",
        "embed_dim", "num_heads", "dropout",
        "p", "inplace",
        "normalized_shape",
        "num_embeddings", "embedding_dim",
    ]:
        if hasattr(module, attr_name):
            val = getattr(module, attr_name)
            if isinstance(val, torch.Tensor):
                attrs[attr_name] = val.shape
            elif isinstance(val, (int, float, bool, str, tuple, list)):
                attrs[attr_name] = val
    return attrs


def _count_params(module: nn.Module) -> int:
    """Count parameters directly owned by this module (not children)."""
    child_params = set()
    for child in module.children():
        for p in child.parameters():
            child_params.add(id(p))
    count = 0
    for p in module.parameters():
        if id(p) not in child_params:
            count += p.numel()
    return count


def _shape_to_list(shape: torch.Size) -> list[int]:
    return list(shape)


def extract_graph(
    model: nn.Module,
    example_input: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    model_name: str | None = None,
) -> ModelGraph:
    """Extract a ModelGraph from a PyTorch model.

    Tries torch.fx symbolic trace first for full dataflow edges.
    Falls back to module hierarchy only if tracing fails.
    """
    if model_name is None:
        model_name = model.__class__.__name__

    graph = ModelGraph(
        model_name=model_name,
        model_class=f"{model.__class__.__module__}.{model.__class__.__qualname__}",
    )

    # Build module hierarchy info
    module_info: dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        module_info[name] = mod

    # Try FX trace for dataflow graph
    try:
        traced = torch.fx.symbolic_trace(model)
        _build_graph_from_fx(graph, traced, model, module_info, example_input)
        logger.info("Successfully extracted graph via torch.fx symbolic trace")
    except Exception as e:
        logger.warning(f"torch.fx symbolic trace failed: {e}. Falling back to module hierarchy.")
        _build_graph_from_hierarchy(graph, model, module_info)

    return graph


def _build_graph_from_fx(
    graph: ModelGraph,
    traced: torch.fx.GraphModule,
    model: nn.Module,
    module_info: dict[str, nn.Module],
    example_input: torch.Tensor | tuple[torch.Tensor, ...] | None,
) -> None:
    """Build graph using torch.fx trace results + module hierarchy."""
    # Step 1: Create module group nodes for the hierarchy
    _create_module_groups(graph, model, module_info)

    # Step 2: Run shape propagation if we have an example input
    node_shapes: dict[str, dict] = {}
    if example_input is not None:
        node_shapes = _propagate_shapes(traced, example_input)

    # Step 3: Create nodes from FX graph
    fx_node_map: dict[str, str] = {}  # fx node name -> our node id

    for fx_node in traced.graph.nodes:
        node_id = fx_node.name
        shapes = node_shapes.get(fx_node.name, {})

        if fx_node.op == "placeholder":
            gn = GraphNode(
                id=node_id,
                name=fx_node.name,
                op="placeholder",
                is_leaf=True,
                parent_id="root",
                input_shapes=shapes.get("input_shapes", []),
                output_shapes=shapes.get("output_shapes", []),
            )
            graph.nodes.append(gn)
            graph.input_nodes.append(node_id)

        elif fx_node.op == "call_module":
            target = fx_node.target
            mod = module_info.get(target)
            parent_id = _get_parent_id(target)
            gn = GraphNode(
                id=node_id,
                name=target.split(".")[-1] if "." in target else target,
                op="call_module",
                is_leaf=True,
                module_type=f"{mod.__class__.__module__}.{mod.__class__.__qualname__}" if mod else None,
                parent_id=parent_id if parent_id in {n.id for n in graph.nodes} else "root",
                attributes=_get_module_attributes(mod) if mod else {},
                param_count=_count_params(mod) if mod else 0,
                input_shapes=shapes.get("input_shapes", []),
                output_shapes=shapes.get("output_shapes", []),
            )
            graph.nodes.append(gn)

        elif fx_node.op in ("call_function", "call_method"):
            func_name = getattr(fx_node.target, "__name__", str(fx_node.target))
            gn = GraphNode(
                id=node_id,
                name=func_name,
                op=fx_node.op,
                is_leaf=True,
                parent_id="root",
                module_type=func_name,
                input_shapes=shapes.get("input_shapes", []),
                output_shapes=shapes.get("output_shapes", []),
            )
            graph.nodes.append(gn)

        elif fx_node.op == "get_attr":
            gn = GraphNode(
                id=node_id,
                name=fx_node.target,
                op="get_attr",
                is_leaf=True,
                parent_id="root",
            )
            graph.nodes.append(gn)

        elif fx_node.op == "output":
            gn = GraphNode(
                id=node_id,
                name="output",
                op="output",
                is_leaf=True,
                parent_id="root",
            )
            graph.nodes.append(gn)
            graph.output_nodes.append(node_id)

        fx_node_map[fx_node.name] = node_id

    # Step 4: Create edges from FX graph connections
    edge_id = 0
    for fx_node in traced.graph.nodes:
        target_id = fx_node_map.get(fx_node.name)
        if target_id is None:
            continue
        for input_node in fx_node.all_input_nodes:
            source_id = fx_node_map.get(input_node.name)
            if source_id is None:
                continue
            edge = GraphEdge(
                id=f"e{edge_id}",
                source=source_id,
                target=target_id,
            )
            graph.edges.append(edge)
            edge_id += 1

    # Step 5: Update children lists on group nodes
    _update_children_lists(graph)


def _create_module_groups(
    graph: ModelGraph,
    model: nn.Module,
    module_info: dict[str, nn.Module],
) -> None:
    """Create module_group nodes for container modules in the hierarchy."""
    # Root group
    root = GraphNode(
        id="root",
        name=model.__class__.__name__,
        op="module_group",
        is_leaf=False,
        module_type=f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        param_count=sum(p.numel() for p in model.parameters()),
    )
    graph.nodes.append(root)
    graph.root_id = "root"

    # Create group nodes for modules that have children
    for name, mod in module_info.items():
        if name == "":  # root, already created
            continue
        children = list(mod.children())
        if len(children) > 0:
            # This is a container module
            parent_id = _get_parent_id(name)
            gn = GraphNode(
                id=name,
                name=name.split(".")[-1],
                op="module_group",
                is_leaf=False,
                module_type=f"{mod.__class__.__module__}.{mod.__class__.__qualname__}",
                parent_id=parent_id if parent_id else "root",
                param_count=sum(p.numel() for p in mod.parameters()),
                attributes=_get_module_attributes(mod),
            )
            graph.nodes.append(gn)


def _build_graph_from_hierarchy(
    graph: ModelGraph,
    model: nn.Module,
    module_info: dict[str, nn.Module],
) -> None:
    """Fallback: build graph from module hierarchy only (no dataflow edges)."""
    _create_module_groups(graph, model, module_info)

    for name, mod in module_info.items():
        if name == "":
            continue
        children = list(mod.children())
        if len(children) == 0:
            # Leaf module
            parent_id = _get_parent_id(name)
            gn = GraphNode(
                id=name,
                name=name.split(".")[-1],
                op="call_module",
                is_leaf=True,
                module_type=f"{mod.__class__.__module__}.{mod.__class__.__qualname__}",
                parent_id=parent_id if parent_id in {n.id for n in graph.nodes} else "root",
                attributes=_get_module_attributes(mod),
                param_count=_count_params(mod),
            )
            graph.nodes.append(gn)

    _update_children_lists(graph)


def _get_parent_id(module_path: str) -> str:
    """Get the parent module path, or 'root' if top-level."""
    parts = module_path.rsplit(".", 1)
    return parts[0] if len(parts) > 1 else "root"


def _update_children_lists(graph: ModelGraph) -> None:
    """Populate children lists on group nodes from parent_id references."""
    node_map = {n.id: n for n in graph.nodes}
    for node in graph.nodes:
        if node.parent_id and node.parent_id in node_map:
            parent = node_map[node.parent_id]
            if node.id not in parent.children:
                parent.children.append(node.id)


def _propagate_shapes(
    traced: torch.fx.GraphModule,
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
) -> dict[str, dict]:
    """Run shape propagation through the traced graph."""
    shapes: dict[str, dict] = {}

    try:
        from torch.fx.passes.shape_prop import ShapeProp
        if isinstance(example_input, tuple):
            ShapeProp(traced).propagate(*example_input)
        else:
            ShapeProp(traced).propagate(example_input)

        for node in traced.graph.nodes:
            meta = node.meta
            if "tensor_meta" in meta:
                tm = meta["tensor_meta"]
                if isinstance(tm, torch.fx.passes.shape_prop.TensorMetadata):
                    shapes[node.name] = {
                        "output_shapes": [[str(tm.dtype), _shape_to_list(tm.shape)]],
                    }
                elif isinstance(tm, (list, tuple)):
                    shapes[node.name] = {
                        "output_shapes": [
                            [str(t.dtype), _shape_to_list(t.shape)]
                            for t in tm
                            if hasattr(t, "dtype")
                        ],
                    }
    except Exception as e:
        logger.warning(f"Shape propagation failed: {e}")

    return shapes
