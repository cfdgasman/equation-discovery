"""Equation discovery with SINDy / PDE-FIND: Lorenz, Burgers, and 2D Navier-Stokes vorticity transport."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp

from discovery import data, pde
from discovery.derivatives import central_difference
from discovery.sparse_regression import format_equation, polynomial_library, stlsq


def lorenz():
    t, X, _ = data.lorenz()
    dt = t[1] - t[0]
    print("Lorenz (SINDy, cubic library of 20 terms, threshold 0.1)")
    print("| noise | sigma | rho | beta | active terms |\n|---|---|---|---|---|")
    rng = np.random.default_rng(0)
    results = {}
    for noise in (0.0, 0.001, 0.01):
        Xn = X + noise * np.std(X) * rng.standard_normal(X.shape)
        if noise:  # smooth before differentiating (Savitzky-Golay-like local polynomial via convolution)
            from scipy.signal import savgol_filter

            dX = savgol_filter(Xn, 41, 4, deriv=1, delta=dt, axis=0)
            Xs = savgol_filter(Xn, 41, 4, axis=0)
        else:
            dX, Xs = central_difference(Xn, dt, axis=0), Xn
        Th, lab = polynomial_library(Xs, 3, ["x", "y", "z"])
        Xi = stlsq(Th, dX, 0.1)
        results[noise] = (Xi, lab)
        sigma = Xi[lab.index("y"), 0]
        rho = Xi[lab.index("x"), 1]
        beta = -Xi[lab.index("z"), 2]
        print(f"| {100 * noise:.1f} % | {sigma:.3f} | {rho:.3f} | {beta:.4f} | {np.count_nonzero(Xi)} (true: 7) |")
    Xi, lab = results[0.0]
    for j, v in enumerate("xyz"):
        print("  " + format_equation(Xi[:, j], lab, f"d{v}/dt", 3))

    def model(_, s):
        th, _ = polynomial_library(s[None, :], 3, ["x", "y", "z"])
        return (th @ Xi)[0]

    sim = solve_ivp(model, (0, 20), X[0], t_eval=t, rtol=1e-10, atol=1e-10).y.T
    fig = plt.figure(figsize=(12, 4.6))
    a1 = fig.add_subplot(1, 3, 1, projection="3d")
    a1.plot(*X.T, lw=0.4, color="C0")
    a1.set_title("Data: Lorenz system")
    a2 = fig.add_subplot(1, 3, 2, projection="3d")
    a2.plot(*sim.T, lw=0.4, color="C3")
    a2.set_title("Simulation of the discovered model")
    for a in (a1, a2):
        a.set_axis_off()
    a3 = fig.add_subplot(1, 3, 3)
    a3.plot(t, X[:, 0], "C0", lw=1.2, label="data")
    a3.plot(t, sim[:, 0], "C3--", lw=1, label="discovered model")
    a3.set(xlabel="t", ylabel="x(t)", xlim=(0, 20), title="Same trajectory until chaos amplifies tiny coefficient errors")
    a3.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("docs/lorenz.png", dpi=110)
    plt.close(fig)


def burgers():
    x, t, u = data.burgers()
    print("\nBurgers' equation (PDE-FIND, 12-term library {1,u,u^2} x {1,u_x,u_xx,u_xxx})")
    print("| noise | discovered equation | error in nu | error in u u_x coeff |\n|---|---|---|---|")
    rng = np.random.default_rng(1)
    rows = []
    for noise in (0.0, 0.001, 0.005, 0.01):
        un = u + noise * np.std(u) * rng.standard_normal(u.shape)
        R, ut, lab = pde.burgers_library(un, x, t, noisy=noise > 0)
        w = pde.identify(R, ut)
        nu = w[lab.index("u_xx")]
        a = w[lab.index("u u_x")]
        rows.append((noise, abs(nu - 0.1) / 0.1, abs(a + 1)))
        print(f"| {100 * noise:.1f} % | `{format_equation(w, lab, 'u_t')}` | {100 * abs(nu - 0.1) / 0.1:.2f} % | {100 * abs(a + 1):.2f} % |")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    T, Xg = np.meshgrid(t, x)
    a1.pcolormesh(Xg, T, u, cmap="viridis", shading="auto")
    a1.set(xlabel="x", ylabel="t", title="Data: u(x, t), viscous Burgers (ν = 0.1)")
    rows = np.array(rows)
    a2.semilogy(100 * rows[1:, 0], 100 * rows[1:, 1], "o-", label="ν")
    a2.semilogy(100 * rows[1:, 0], 100 * rows[1:, 2], "s-", label="u u_x coefficient")
    a2.set(xlabel="noise (% of std)", ylabel="relative error (%)", title="Robustness to measurement noise")
    a2.grid(alpha=0.3, which="both")
    a2.legend()
    fig.tight_layout()
    fig.savefig("docs/burgers.png", dpi=120)
    plt.close(fig)


def navier_stokes():
    x, t, W, U, V = data.vorticity_2d()
    print("\n2D Navier-Stokes, vorticity form (PDE-FIND, 24-term library {1,w,u,v} x {1,w_x,w_y,w_xx,w_xy,w_yy})")
    print("true:  w_t = -u w_x - v w_y + 0.005 (w_xx + w_yy)")
    R, wt, lab = pde.vorticity_library(W, U, V, x, t)
    w = pde.identify(R, wt)
    print("found: " + format_equation(w, lab, "w_t", 5))
    coef = {name: c for name, c in zip(lab, w)}
    nu_err = max(abs(coef["w_xx"] - 0.005), abs(coef["w_yy"] - 0.005)) / 0.005
    print(f"viscosity error: {100 * nu_err:.2f} %, advection coefficients: {coef['u w_x']:.5f}, {coef['v w_y']:.5f}")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    order = np.argsort(lab)
    names = [lab[i] for i in order]
    true = {"u w_x": -1, "v w_y": -1, "w_xx": 0.005, "w_yy": 0.005}
    ax.bar(np.arange(len(lab)) - 0.2, [true.get(n, 0) for n in names], 0.4, label="true", color="grey")
    ax.bar(np.arange(len(lab)) + 0.2, [w[i] for i in order], 0.4, label="discovered", color="C3")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xticks(np.arange(len(lab)), names, rotation=70, fontsize=7)
    ax.set(ylabel="coefficient", title="Candidate terms for ∂ω/∂t: sparse regression keeps exactly four")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("docs/ns_coefficients.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    fig.subplots_adjust(0, 0, 1, 0.92)
    m = np.abs(W).max()
    im = ax.imshow(W[..., 0].T, origin="lower", cmap="RdBu_r", vmin=-m, vmax=m, extent=(0, 2 * np.pi, 0, 2 * np.pi))
    ax.set(xticks=[], yticks=[])
    title = ax.set_title("")

    def update(k):
        im.set_data(W[..., k].T)
        title.set_text(f"Vorticity data, t = {t[k]:.1f}")
        return [im]

    FuncAnimation(fig, update, frames=W.shape[2]).save("docs/vorticity.gif", writer=PillowWriter(fps=12), dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    lorenz()
    burgers()
    navier_stokes()
