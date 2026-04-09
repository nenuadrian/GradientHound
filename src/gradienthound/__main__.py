"""``python -m gradienthound`` — launch the standalone Dash dashboard."""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gradienthound",
        description="Launch the GradientHound dashboard",
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="Port to serve the dashboard on (default: 8050)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to a .gh.json model export file or folder to recursively search for model exports",
    )
    parser.add_argument(
        "--checkpoints", nargs="+", default=None,
        help="Checkpoint files or folders to compare (.pt, .pth, .ckpt). Folders are recursively searched.",
    )
    parser.add_argument(
        "--loader", type=str, default=None,
        help="Path to a Python script defining load_checkpoint(path) -> nn.Module",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None,
        help="Weights & Biases entity (team or username)",
    )
    parser.add_argument(
        "--wandb-project-run-id", type=str, default=None,
        help="Weights & Biases project/run_id, e.g. my-project/abc123",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run Dash in debug mode with hot-reload",
    )
    args = parser.parse_args()

    if not (1 <= args.port <= 65535):
        parser.error(f"--port must be between 1 and 65535, got {args.port}")

    from gradienthound._dashboard import create_app
    from gradienthound.checkpoint import discover_checkpoints, discover_model_exports

    # Discover checkpoint files from provided locations (files or directories)
    checkpoint_paths = args.checkpoints
    if checkpoint_paths:
        checkpoint_paths = discover_checkpoints(checkpoint_paths)

    # Discover model export files from provided location (file or directory)
    model_paths = None
    if args.model:
        model_paths = discover_model_exports([args.model])

    app = create_app(
        model_paths=model_paths,
        checkpoint_paths=checkpoint_paths,
        loader_path=args.loader,
        wandb_entity=args.wandb_entity,
        wandb_project_run_id=args.wandb_project_run_id,
    )
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
