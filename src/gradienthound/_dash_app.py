"""Backwards-compatible re-export — use gradienthound._dashboard instead."""
from gradienthound._dashboard._app import create_app

__all__ = ["create_app"]
