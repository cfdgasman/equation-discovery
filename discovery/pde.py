"""PDE-FIND pipelines: build candidate libraries from data and identify the governing PDE."""

from __future__ import annotations

import numpy as np

from .derivatives import central_difference, poly_derivative, spectral
from .sparse_regression import stridge


def burgers_library(u, x, t, noisy=False, width=6, deg=5, time_width=6):
    """Library {1, u, u^2} x {1, u_x, u_xx, u_xxx} and the target u_t, flattened over (x, t).
    Clean data: spectral x-derivatives and central differences in t.
    Noisy data: local Chebyshev polynomial fits in both x and t (Rudy et al. 2017)."""
    dx = x[1] - x[0]
    if not noisy:
        ut = central_difference(u, t[1] - t[0], axis=1)
        d = [spectral(u, dx, o, axis=0) for o in (1, 2, 3)]
        U = u
    else:
        nx, nt = u.shape
        ut_full = np.array([poly_derivative(u[i], t, deg, time_width, (1,))[0] for i in range(nx)])
        U = u[:, time_width:nt - time_width]
        ut = ut_full
        derivs = np.array([poly_derivative(u[:, j], x, deg, width, (1, 2, 3)) for j in range(time_width, nt - time_width)])
        d = [np.pad(derivs[:, i, :].T, ((width, width), (0, 0))) for i in range(3)]
        keep = slice(width, nx - width)
        U, ut, d = U[keep], ut[keep], [di[keep] for di in d]
    cols, labels = [], []
    for p, pname in ((0, ""), (1, "u"), (2, "u^2")):
        for dd, dname in ((None, ""), (d[0], "u_x"), (d[1], "u_xx"), (d[2], "u_xxx")):
            base = U**p
            col = base if dd is None else base * dd
            name = (pname + (" " if pname and dname else "") + dname) or "1"
            cols.append(col.ravel())
            labels.append(name)
    return np.column_stack(cols), ut.ravel(), labels


def vorticity_library(W, U, V, x, t, n_samples=40_000, seed=0, noise=0.0):
    """Library {1, w, u, v} x {1, w_x, w_y, w_xx, w_xy, w_yy} evaluated at random space-time points."""
    rng = np.random.default_rng(seed)
    dx = x[1] - x[0]
    if noise:
        W = W + noise * np.std(W) * rng.standard_normal(W.shape)
        U = U + noise * np.std(U) * rng.standard_normal(U.shape)
        V = V + noise * np.std(V) * rng.standard_normal(V.shape)
    wt = central_difference(W, t[1] - t[0], axis=2)
    wx, wy = spectral(W, dx, 1, 0), spectral(W, dx, 1, 1)
    wxx, wyy = spectral(W, dx, 2, 0), spectral(W, dx, 2, 1)
    wxy = spectral(wx, dx, 1, 1)
    n = W.shape[0]
    ti = rng.integers(2, W.shape[2] - 2, n_samples)
    ii = rng.integers(0, n, n_samples)
    jj = rng.integers(0, n, n_samples)
    pick = lambda a: a[ii, jj, ti]  # noqa: E731
    base = {"1": np.ones(n_samples), "w": pick(W), "u": pick(U), "v": pick(V)}
    ders = {"": None, "w_x": pick(wx), "w_y": pick(wy), "w_xx": pick(wxx), "w_xy": pick(wxy), "w_yy": pick(wyy)}
    cols, labels = [], []
    for bn, b in base.items():
        for dn, d in ders.items():
            if bn == "1" and dn == "":
                cols.append(b)
                labels.append("1")
            elif dn == "":
                cols.append(b)
                labels.append(bn)
            else:
                cols.append(b * d)
                labels.append(dn if bn == "1" else f"{bn} {dn}")
    return np.column_stack(cols), pick(wt), labels


def identify(R, ut, **kw):
    return stridge(R, ut, **kw)
