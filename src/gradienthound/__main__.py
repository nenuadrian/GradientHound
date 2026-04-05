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
        "--data-dir", type=str, default=None,
        help="Path to an IPC data directory to load",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to a .gh.json model export file",
    )
    parser.add_argument(
        "--checkpoints", nargs="+", default=None,
        help="Checkpoint files to compare (.pt, .pth, .ckpt)",
    )
    parser.add_argument(
        "--loader", type=str, default=None,
        help="Path to a Python script defining load_checkpoint(path) -> nn.Module",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run Dash in debug mode with hot-reload",
    )
    args = parser.parse_args()

    from gradienthound._dash_app import create_app

    app = create_app(
        data_dir=args.data_dir,
        model_path=args.model,
        checkpoint_paths=args.checkpoints,
        loader_path=args.loader,
    )
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
