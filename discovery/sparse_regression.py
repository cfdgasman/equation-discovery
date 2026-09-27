"""Sparse regression for equation discovery.

* STLSQ  - sequentially thresholded least squares (SINDy; Brunton, Proctor & Kutz, PNAS 2016)
* STRidge - sequentially thresholded ridge regression with a tolerance search (PDE-FIND;
            Rudy, Brunton, Proctor & Kutz, Science Advances 2017)

Both solve  Theta xi ~= dX  for a sparse coefficient vector xi, where the columns of the library
Theta are candidate terms (monomials, derivatives, products) evaluated on data.
"""

from __future__ import annotations

import itertools

import numpy as np


def stlsq(Theta, dX, threshold, max_iter=20):
    """Sequentially thresholded least squares, one column of dX at a time."""
    Xi = np.linalg.lstsq(Theta, dX, rcond=None)[0]
    for _ in range(max_iter):
        small = np.abs(Xi) < threshold
        Xi[small] = 0
        for j in range(dX.shape[1]):
            big = ~small[:, j]
            if big.any():
                Xi[big, j] = np.linalg.lstsq(Theta[:, big], dX[:, j], rcond=None)[0]
    return Xi


def polynomial_library(X, degree, names=None):
    """All monomials of the state variables up to `degree` (including the constant)."""
    n = X.shape[1]
    names = names or [f"x{i}" for i in range(n)]
    cols, labels = [np.ones(len(X))], ["1"]
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n), d):
            cols.append(np.prod(X[:, combo], axis=1))
            labels.append("".join(names[i] for i in combo))
    return np.column_stack(cols), labels


def _ridge(A, b, lam):
    if lam == 0:
        return np.linalg.lstsq(A, b, rcond=None)[0]
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)


def stridge_once(R, ut, lam, tol, max_iter=10):
    """One STRidge pass with fixed hard threshold `tol` on the (normalised) coefficients."""
    w = _ridge(R, ut, lam)
    big = np.ones(len(w), dtype=bool)
    for _ in range(max_iter):
        new_big = np.abs(w) >= tol
        if new_big.sum() == 0:
            return np.zeros_like(w)
        if np.array_equal(new_big, big):
            break
        big = new_big
        w = np.zeros_like(w)
        w[big] = _ridge(R[:, big], ut, lam)
    w_final = np.zeros_like(w)
    w_final[big] = np.linalg.lstsq(R[:, big], ut, rcond=None)[0]  # debias on the support
    return w_final


def stridge(R, ut, lam=1e-5, d_tol=None, n_tol=40, l0_penalty=None, split=0.8, seed=0):
    """STRidge with a search over the threshold, selecting by held-out error + l0 penalty (Rudy et al. 2017).
    Columns are normalised internally; coefficients are returned in the original units."""
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(R, axis=0)
    Rn = R / norms
    n = len(ut)
    idx = rng.permutation(n)
    tr, te = idx[: int(split * n)], idx[int(split * n):]
    w_best = _ridge(Rn[tr], ut[tr], lam)
    if l0_penalty is None:
        l0_penalty = 1e-3 * np.linalg.cond(Rn)
    def score(w):
        return np.linalg.norm(ut[te] - Rn[te] @ w) ** 2 + l0_penalty * np.count_nonzero(w)

    err_best = score(w_best)
    tol = d_tol if d_tol is not None else 1e-3 * np.abs(w_best).max()  # start small, grow while it helps
    for _ in range(n_tol):
        w = stridge_once(Rn[tr], ut[tr], lam, tol)
        err = score(w)
        if err <= err_best:
            err_best, w_best = err, w
            tol *= 1.5
        else:
            tol *= 1.1
    return w_best / norms


def format_equation(w, labels, lhs, digits=4, eps=1e-12):
    terms = [f"{c:+.{digits}f} {name}" for c, name in zip(w, labels) if abs(c) > eps]
    return f"{lhs} = " + (" ".join(terms) if terms else "0")
