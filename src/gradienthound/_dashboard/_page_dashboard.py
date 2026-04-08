"""Dashboard page — backward compatibility re-exports.

The original monolithic dashboard has been split into focused pages:
  _page_overview.py      — Overview (stat cards, I/O, health map)
  _page_architecture.py  — Architecture (FX graph, module tree, live analysis)
  _page_weight_health.py — Weight Health (norms, anomalies, health table)
  _page_distributions.py — Distributions (histograms, kurtosis, entropy)
  _page_spectral.py      — Spectral (effective rank, SVD, ESD, WeightWatcher)
  _page_dynamics.py      — Training Dynamics (drift, velocity, convergence)
"""
from __future__ import annotations

# Re-export for any code that still imports from this module
from ._page_overview import overview_page as dashboard_page  # noqa: F401
from ._page_overview import landing_page_empty  # noqa: F401
