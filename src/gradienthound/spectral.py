"""Spectral analysis of weight matrices (WeightWatcher-style metrics).

Computes per-layer metrics from the Empirical Spectral Density (ESD) of
weight matrices — the distribution of eigenvalues of W^T W.  These are the
same diagnostics that the WeightWatcher library computes, implemented here
directly from singular values so we can run on raw tensors without needing
a live ``nn.Module``.

Key metrics:

* **alpha** — power-law exponent of the ESD tail.  Well-trained layers
  typically have alpha in [2, 6].  Values below 2 suggest heavy-tailed /
  strongly correlated weights; values above 6 suggest undertrained or
  near-random layers.
* **alpha_weighted** — ``alpha * log10(lambda_max)``, combining shape and
  scale of the ESD into a single quality score.
* **log_spectral_norm** — ``log10(lambda_max)``, the log of the largest
  eigenvalue.
* **mp_softrank** — ratio of the Marchenko-Pastur (random bulk) edge to the
  actual maximum eigenvalue.  Near 1.0 means the layer is close to random;
  near 0.0 means strong learned structure dominates.
* **num_spikes** — count of eigenvalues above the MP bulk edge, i.e. the
  number of "informative" directions learned beyond random noise.

References:
    Martin & Mahoney, "Implicit Self-Regularization in Deep Neural Networks",
    arXiv:1901.08276.
"""
from __future__ import annotations

import math
from typing import Any


def compute_spectral_metrics(
    singular_values: list[float],
    matrix_shape: tuple[int, int],
    *,
    min_evals: int = 50,
) -> dict[str, Any]:
    """Compute WeightWatcher-style spectral metrics from singular values.

    Args:
        singular_values: Singular values of the weight matrix (descending).
        matrix_shape: ``(rows, cols)`` of the original weight matrix.
        min_evals: Minimum number of eigenvalues required for a reliable
            power-law fit.  Layers with fewer are skipped.

    Returns:
        Dict of spectral metrics, or empty dict if the layer is too small
        or the fit fails.
    """
    if len(singular_values) < min_evals:
        return {}

    evals = [sv * sv for sv in singular_values]
    lambda_max = evals[0] if evals else 0.0

    if lambda_max < 1e-30:
        return {}

    result: dict[str, Any] = {}

    log_spectral_norm = math.log10(max(lambda_max, 1e-30))
    result["log_spectral_norm"] = log_spectral_norm

    # ── Power-law fit via the powerlaw library ──────────────────────
    alpha = _fit_power_law(evals)
    if alpha is not None:
        result["alpha"] = alpha
        result["alpha_weighted"] = alpha * log_spectral_norm

    # ── Marchenko-Pastur bulk edge and spike count ──────────────────
    mp = _mp_metrics(evals, matrix_shape)
    result.update(mp)

    # ── Store the ESD for optional visualization ────────────────────
    result["esd"] = evals

    return result


def _fit_power_law(evals: list[float]) -> float | None:
    """Fit a power-law to the eigenvalue distribution, return alpha."""
    try:
        import powerlaw
    except ImportError:
        return None

    # Filter to positive eigenvalues
    pos_evals = [e for e in evals if e > 0]
    if len(pos_evals) < 10:
        return None

    try:
        fit = powerlaw.Fit(pos_evals, verbose=False)
        alpha = float(fit.alpha)
        # Sanity: alpha should be positive and finite
        if not math.isfinite(alpha) or alpha <= 0:
            return None
        return alpha
    except Exception:
        return None


def _mp_metrics(
    evals: list[float],
    matrix_shape: tuple[int, int],
) -> dict[str, Any]:
    """Compute Marchenko-Pastur edge, softrank, and spike count.

    The MP distribution describes the eigenvalue distribution of a random
    matrix with i.i.d. entries.  Eigenvalues above the MP upper edge
    ``lambda_+`` represent learned structure (signal), while eigenvalues
    below it are indistinguishable from noise.
    """
    rows, cols = matrix_shape
    n = max(rows, cols)
    m = min(rows, cols)

    if m < 2 or n < 2:
        return {}

    q = m / n  # aspect ratio, <= 1

    # Estimate the bulk eigenvalue mean, trimming the top 10% to exclude
    # spike eigenvalues that would bias the estimate upward.
    pos_evals = sorted([e for e in evals if e > 0])
    if len(pos_evals) < 4:
        return {}

    trim = max(1, len(pos_evals) // 10)
    bulk_evals = pos_evals[:-trim] if trim < len(pos_evals) else pos_evals
    bulk_mean = sum(bulk_evals) / len(bulk_evals)

    if bulk_mean < 1e-30:
        return {}

    # For W of shape (n, m), the eigenvalues of W^T W from a random
    # matrix with i.i.d. entries of variance sigma^2 have:
    #   mean(evals) ≈ n * sigma^2
    #   MP upper edge = n * sigma^2 * (1 + sqrt(m/n))^2
    #                 = bulk_mean * (1 + sqrt(q))^2
    lambda_plus = bulk_mean * (1 + math.sqrt(q)) ** 2

    lambda_max = max(evals) if evals else 0.0

    mp_softrank = lambda_plus / max(lambda_max, 1e-30)
    mp_softrank = min(mp_softrank, 1.0)  # clamp at 1.0

    num_spikes = sum(1 for e in evals if e > lambda_plus)

    return {
        "mp_softrank": mp_softrank,
        "num_spikes": num_spikes,
        "lambda_plus": lambda_plus,
    }
