# GradientHound

<p align="center">
    <img src="https://github.com/nenuadrian/GradientHound/blob/main/docs/assets/logo.png?raw=true" alt="logo" />
</p>

Mainly post-training tooling inspect model architectures, gradients, weights, and optimizer state. With hooks that can be added to existing training frameworks to capture model configurations. Expectations that regular checkpoints are created and are made available when running GH.

This is mainly designed to work with my framework of training and my needs, and it is unlikely to work as is for anyone else in its current form, but it might be something useful to have around, bringing together multiple tools for network introspection, and hopefully develop over time a holistic analysis framework that can adapt to any model.

## Install

```bash
pip install gradienthound            # core training capture + model export
pip install gradienthound[torch]     # + PyTorch integration
pip install gradienthound[dash]      # + standalone Dash dashboard
```

## Live training capture

GradientHound provides a wandb-style API that hooks into your training loop and captures rich telemetry.

### Quick start

```python
import torch
import torch.nn as nn
import gradienthound

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
    nn.Flatten(), nn.Linear(32 * 32 * 32, 10),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Initialize capture for this run
gradienthound.init(metadata={"lr": 1e-3, "batch_size": 32})

# Register model + optimizer for visualization
gradienthound.register_model("mymodel", model)
gradienthound.register_optimizer("adam", optimizer)

# Enable automatic gradient/weight capture via PyTorch hooks
gradienthound.watch(model, name="mymodel")

for epoch in range(100):
    out = model(torch.randn(8, 3, 32, 32))
    loss = nn.functional.cross_entropy(out, torch.randint(0, 10, (8,)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Flush stats to the dashboard
    gradienthound.step()

gradienthound.shutdown()
```

### API reference

| Function | Description |
|---|---|
| `gradienthound.init(metadata=None)` | Initialize a run and start telemetry capture. |
| `gradienthound.register_model(name, model)` | Register an `nn.Module` for architecture visualization. |
| `gradienthound.register_optimizer(name, optimizer)` | Register an optimizer for state inspection. |
| `gradienthound.watch(model, name, log_gradients=True, log_activations=False, weight_every=50)` | Attach PyTorch hooks for automatic gradient/activation capture. |
| `gradienthound.step(step=None)` | Flush buffered stats to the dashboard. Call once per training step. |
| `gradienthound.log_weights(name=None)` | Force an immediate weight snapshot. |
| `gradienthound.log_attention(name, weights)` | Log an attention weight matrix for heatmap visualization. |
| `gradienthound.log_predictions(predicted, actual, name="default")` | Log predicted vs actual values for calibration plots. |
| `gradienthound.capture_wandb()` | Monkey-patch `wandb.log()` to also capture scalars in GradientHound. |
| `gradienthound.shutdown()` | Clean up hooks and close the active capture run. |

### Dashboard pages

The live UI provides eight pages, each auto-refreshing during training:

- **Dashboard** -- run overview with model/optimizer summaries
- **Metrics** -- live charts for any scalars logged via wandb or directly
- **Architecture** -- interactive model graph with health overlays (gradient flow, activation sparsity, weight structure)
- **Weights** -- per-layer histograms, SVD analysis, heatmaps, norms
- **Gradients** -- gradient flow, cosine similarity, update ratios, dead neurons
- **Training** -- prediction scatter plots, CKA similarity, attention patterns
- **Optimizers** -- config, parameter groups, state buffers, warmup progress
- **Network State** -- raw layer-by-layer parameter values

### Integration patterns

**Minimal:**

```python
gh = gradienthound.init()
gradienthound.register_model("net", model)
gradienthound.watch(model, "net")
# ... train ...
gradienthound.step()
```

**With wandb:**

```python
import wandb
wandb.init(project="my-project")

gradienthound.init()
gradienthound.capture_wandb()  # auto-captures wandb.log() scalars
```

**Framework integration (base trainer pattern):**

If your project has a base trainer class, add GradientHound once in the base and all trainers inherit it:

```python
class BaseTrainer:
    def __init__(self, seed, device):
        ...

    def register_model(self, name, model, *, step=0):
        if gradienthound is not None:
            gradienthound.register_model(name, model)
            gradienthound.watch(model, name=name)

    def register_optimizer(self, name, optimizer):
        if gradienthound is not None:
            gradienthound.register_optimizer(name, optimizer)
```

## Model export

Export a model's full computation graph to a JSON file for offline visualization. Uses `torch.export` to capture the ATen-level FX graph with per-op shapes, dtypes, and dataflow edges.

```python
import torch
import gradienthound

model = MyModel()
sample_input = (torch.randn(1, 3, 224, 224),)

# Export to JSON (no weights saved, only metadata)
gradienthound.export_model(model, sample_input, "model.gh.json")
```

The exported `.gh.json` file contains:

- **Module tree** -- hierarchical nn.Module structure with per-layer parameter counts and attributes
- **FX computation graph** -- every ATen op with input/output shapes, dtypes, and which nn.Module it belongs to
- **Dataflow edges** -- actual tensor flow between ops, including skip connections and branching
- **Parameter metadata** -- shape, dtype, device, requires_grad for every parameter and buffer
- **Graph signature** -- which inputs are parameters, buffers, or user inputs

```python
gradienthound.export_model(
    model,
    example_inputs,         # tuple of tensors, same as model(*example_inputs)
    output="model.gh.json", # path to write (None to just return the dict)
    dynamic_shapes=None,    # optional, passed to torch.export
    strict=True,            # set False for models with data-dependent control flow
)
```

If `torch.export` fails (unsupported ops, dynamic control flow), the export falls back to module-tree only with a warning.

## Standalone dashboard

Browse exported models or saved training data without an active training process:

```bash
python -m gradienthound --model model.gh.json  # load an exported model
python -m gradienthound --model ./checkpoints/ # load first .gh.json from a directory
python -m gradienthound --port 9000            # custom port
python -m gradienthound --data-dir ./run_data  # load IPC data directory
python -m gradienthound --debug                # Dash hot-reload
```

The dashboard shows an interactive computation graph (powered by Cytoscape.js) with:
- Nodes colored by module type (conv, linear, norm, activation, etc.)
- Directed edges showing actual dataflow including skip connections
- Click-to-inspect detail panel for each op
- Zoom, pan, and drag support

When a full FX computation graph is available (from `torch.export`), it displays every ATen op with shapes and module mapping. When only the module tree is available (export fallback), it displays the layer sequence instead.

## License

MIT
