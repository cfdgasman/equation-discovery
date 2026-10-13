import numpy as np
import pytest

from discovery import data, pde
from discovery.derivatives import central_difference, spectral
from discovery.sparse_regression import polynomial_library, stlsq


def test_spectral_derivative_is_exact_for_fourier_modes():
    x = 2 * np.pi * np.arange(64) / 64
    assert np.allclose(spectral(np.sin(3 * x), x[1] - x[0], 2), -9 * np.sin(3 * x), atol=1e-10)


def test_stlsq_recovers_sparse_linear_model():
    rng = np.random.default_rng(0)
    Th = rng.standard_normal((500, 8))
    xi = np.array([0, 2.0, 0, 0, -3.0, 0, 0, 0.5])
    got = stlsq(Th, (Th @ xi)[:, None], 0.1)[:, 0]
    assert np.allclose(got, xi, atol=1e-10)


def test_sindy_lorenz():
    t, X, _ = data.lorenz(t_end=10.0)
    Th, lab = polynomial_library(X, 2, ["x", "y", "z"])
    Xi = stlsq(Th, central_difference(X, t[1] - t[0], axis=0), 0.1)
    assert np.count_nonzero(Xi) == 7
    assert Xi[lab.index("y"), 0] == pytest.approx(10, rel=1e-3)
    assert Xi[lab.index("x"), 1] == pytest.approx(28, rel=1e-3)
    assert Xi[lab.index("z"), 2] == pytest.approx(-8 / 3, rel=1e-3)


def test_pde_find_burgers():
    x, t, u = data.burgers(n=128, t_end=5.0, n_t=51)
    R, ut, lab = pde.burgers_library(u, x, t)
    w = pde.identify(R, ut)
    active = {lab[i] for i in np.nonzero(w)[0]}
    assert active == {"u u_x", "u_xx"}
    assert w[lab.index("u_xx")] == pytest.approx(0.1, rel=0.02)


def test_pde_find_vorticity_equation():
    x, t, W, U, V = data.vorticity_2d(n=48, t_end=2.0, n_t=41)
    R, wt, lab = pde.vorticity_library(W, U, V, x, t, n_samples=20_000)
    w = pde.identify(R, wt)
    active = {lab[i] for i in np.nonzero(w)[0]}
    assert active == {"u w_x", "v w_y", "w_xx", "w_yy"}
    assert w[lab.index("w_xx")] == pytest.approx(0.005, rel=0.05)
