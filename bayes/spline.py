"""Natural cubic spline basis, used for the aging curve.

A natural spline is constrained to be linear beyond the outer knots, which is
the property that matters here: the panel has very few 21-year-olds and very
few 39-year-olds, and an unconstrained cubic would take those handful of rows
as licence to bend sharply at both ends of the aging curve.
"""

from __future__ import annotations

import numpy as np


def natural_spline_basis(
    x: np.ndarray, knots: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Basis for a natural cubic spline, excluding the intercept.

    Returns ``(basis, spec)`` with ``basis`` of shape (n, len(knots) - 1). The
    spec carries the knots and the centring/scaling applied, so the identical
    basis can be rebuilt for new ages at prediction time - a basis refit to the
    prediction rows would not mean the same thing as the fitted coefficients.
    """
    x = np.asarray(x, float)
    knots = np.asarray(knots, float)
    k = len(knots)
    if k < 3:
        raise ValueError("need at least 3 knots")

    def cubic_plus(v):
        return np.where(v > 0, v ** 3, 0.0)

    def d(j):
        return (cubic_plus(x - knots[j]) - cubic_plus(x - knots[-1])) / (
            knots[-1] - knots[j]
        )

    cols = [x] + [d(j) - d(k - 2) for j in range(k - 2)]
    basis = np.column_stack(cols)

    # Centre and scale so every column is on a comparable footing and a single
    # prior width is sensible for all of them. Centring also keeps the position
    # intercept interpretable as the level at the average age.
    center = basis.mean(axis=0)
    scale = basis.std(axis=0)
    scale[scale == 0] = 1.0
    return (basis - center) / scale, {
        "knots": knots,
        "center": center,
        "scale": scale,
    }


def apply_spline_basis(x: np.ndarray, spec: dict) -> np.ndarray:
    """Rebuild the basis of :func:`natural_spline_basis` for new points."""
    x = np.asarray(x, float)
    knots = spec["knots"]
    k = len(knots)

    def cubic_plus(v):
        return np.where(v > 0, v ** 3, 0.0)

    def d(j):
        return (cubic_plus(x - knots[j]) - cubic_plus(x - knots[-1])) / (
            knots[-1] - knots[j]
        )

    basis = np.column_stack([x] + [d(j) - d(k - 2) for j in range(k - 2)])
    return (basis - spec["center"]) / spec["scale"]
