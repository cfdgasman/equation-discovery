"""Derivative estimation from data: spectral, finite differences, and local polynomial fits (for noise)."""

from __future__ import annotations

import numpy as np


def spectral(u, dx, order=1, axis=0):
    n = u.shape[axis]
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    shape = [1] * u.ndim
    shape[axis] = n
    return np.fft.ifft((1j * k.reshape(shape)) ** order * np.fft.fft(u, axis=axis), axis=axis).real


def central_difference(u, dt, axis=-1):
    """Second-order central differences (one-sided at the ends)."""
    return np.gradient(u, dt, axis=axis, edge_order=2)


def poly_derivative(u, x, deg=4, width=5, orders=(1, 2)):
    """Derivatives of noisy 1D data by fitting a polynomial of degree `deg` on a window of
    2*width+1 points (Rudy et al. 2017). Returns an array (len(orders), len(u) - 2*width)."""
    n = len(u)
    out = np.zeros((len(orders), n - 2 * width))
    for j in range(width, n - width):
        pts = slice(j - width, j + width + 1)
        p = np.polynomial.chebyshev.Chebyshev.fit(x[pts], u[pts], deg)
        for i, o in enumerate(orders):
            out[i, j - width] = p.deriv(o)(x[j])
    return out
