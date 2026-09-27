"""Data generators: Lorenz system, 1D viscous Burgers, 2D Navier-Stokes (vorticity form)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def lorenz(t_end=20.0, dt=0.002, x0=(-8.0, 8.0, 27.0), sigma=10.0, rho=28.0, beta=8 / 3):
    def f(t, X):
        x, y, z = X
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    t = np.arange(0, t_end, dt)
    sol = solve_ivp(f, (0, t_end), x0, t_eval=t, rtol=1e-12, atol=1e-12)
    X = sol.y.T
    dX = np.array([f(0, x) for x in X])  # exact derivatives, for reference
    return t, X, dX


def burgers(nu=0.1, n=256, L=16.0, t_end=10.0, n_t=101):
    """u_t + u u_x = nu u_xx on a periodic domain [-L/2, L/2), Gaussian initial data.
    Fourier pseudo-spectral in space (2/3 dealiasing), adaptive RK45 in time."""
    x = -L / 2 + L * np.arange(n) / n
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    dealias = np.abs(k) < (2 / 3) * np.abs(k).max()

    def rhs(t, u):
        uh = np.fft.fft(u)
        ux = np.fft.ifft(1j * k * uh * dealias).real
        uxx = np.fft.ifft(-(k**2) * uh).real
        return -u * ux + nu * uxx

    t = np.linspace(0, t_end, n_t)
    sol = solve_ivp(rhs, (0, t_end), np.exp(-((x + 2) ** 2)), t_eval=t, method="RK45", rtol=1e-10, atol=1e-12)
    return x, t, sol.y  # u[x, t]


def vorticity_2d(nu=0.005, n=128, t_end=8.0, n_t=81, seed=2):
    """2D incompressible Navier-Stokes in vorticity form on [0, 2pi)^2, pseudo-spectral + RK4:
        w_t + u w_x + v w_y = nu lap(w),   u = psi_y, v = -psi_x,  lap(psi) = -w.
    Returns grid, times, and w, u, v arrays of shape (n, n, n_t)."""
    rng = np.random.default_rng(seed)
    L = 2 * np.pi
    x = L * np.arange(n) / n
    k = np.fft.fftfreq(n, d=1 / n)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX**2 + KY**2
    K2i = np.where(K2 == 0, 0, 1 / np.where(K2 == 0, 1, K2))
    dealias = (np.abs(KX) < n / 3) & (np.abs(KY) < n / 3)
    # random smooth initial vorticity (energy near |k| ~ 4)
    kk = np.sqrt(K2)
    amp = np.where((kk > 0), kk**4 * np.exp(-(kk / 4) ** 2), 0)
    wh = amp * np.exp(2j * np.pi * rng.random((n, n)))
    w = np.fft.ifft2(wh).real
    w *= 3 / np.abs(w).max()
    wh = np.fft.fft2(w)

    def vel(wh):
        psih = wh * K2i
        return np.fft.ifft2(1j * KY * psih).real, np.fft.ifft2(-1j * KX * psih).real

    def rhs(wh):
        u, v = vel(wh)
        wx = np.fft.ifft2(1j * KX * wh).real
        wy = np.fft.ifft2(1j * KY * wh).real
        return -np.fft.fft2(u * wx + v * wy) * dealias - nu * K2 * wh

    dt = 2e-3
    times = np.linspace(0, t_end, n_t)
    W = np.empty((n, n, n_t))
    U = np.empty_like(W)
    V = np.empty_like(W)
    t = 0.0
    for j, target in enumerate(times):
        while t < target - 1e-12:
            h = min(dt, target - t)
            k1 = rhs(wh)
            k2 = rhs(wh + 0.5 * h * k1)
            k3 = rhs(wh + 0.5 * h * k2)
            k4 = rhs(wh + h * k3)
            wh = wh + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        W[..., j] = np.fft.ifft2(wh).real
        U[..., j], V[..., j] = vel(wh)
    return x, times, W, U, V
