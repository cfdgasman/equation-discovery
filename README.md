# Data-Driven Equation Discovery: SINDy and PDE-FIND

[![CI](https://github.com/cfdgasman/equation-discovery/actions/workflows/ci.yml/badge.svg)](https://github.com/cfdgasman/equation-discovery/actions/workflows/ci.yml)

Governing equations discovered **from data alone** by sparse regression, following Brunton, Proctor & Kutz (**SINDy**, PNAS 2016) and Rudy, Brunton, Proctor & Kutz (**PDE-FIND**, Science Advances 2017). The main demonstration is fluid dynamics: from snapshots of a 2D Navier–Stokes simulation, the method recovers the **vorticity transport equation** and the **viscosity**.

<p align="center">
<img src="docs/vorticity.gif" width="300" alt="Vorticity data">
</p>

```
true   : ω_t = −u ω_x − v ω_y + 0.005 (ω_xx + ω_yy)
found  : ω_t = −0.99986 u ω_x − 0.99980 v ω_y + 0.00500 ω_xx + 0.00500 ω_yy
```

## Method

Suppose the dynamics are **u**<sub>t</sub> = N(**u**, **u**<sub>x</sub>, **u**<sub>xx</sub>, …) and N is a sparse combination of simple candidate terms. The problem then becomes a linear regression:

$$ \mathbf u_t = \Theta(\mathbf u)\,\boldsymbol\xi,\qquad \Theta = \big[\,1,\ u,\ u^2,\ u_x,\ u\,u_x,\ u_{xx},\ \dots\big],\qquad \boldsymbol\xi \text{ sparse}. $$

1. **Measure** u at many points in space and time.
2. **Differentiate.** The code uses spectral derivatives for periodic clean data and central differences in time. For noisy data it uses local Chebyshev-polynomial fits or Savitzky–Golay filters.
3. **Build the library** Θ: every column is a candidate term evaluated at every sample point.
4. **Sparse regression.**
   - **STLSQ** (SINDy): least squares, zero out the coefficients below a threshold, refit on the survivors, repeat.
   - **STRidge** (PDE-FIND): the same with ridge regularisation and normalised columns. The threshold is chosen automatically by minimising held-out error + λ‖ξ‖₀.

The result is an interpretable equation, not a black box.

## Results

### 1. Lorenz system (SINDy, 20 candidate terms up to cubic)

| noise | σ (10) | ρ (28) | β (8/3) | active terms |
|---|---|---|---|---|
| 0 % | 9.999 | 27.992 | 2.6664 | 7 = true |
| 0.1 % | 9.999 | 27.967 | 2.6655 | 7 = true |
| 1 % | 9.990 | 28.005 | 2.6661 | 7 = true |

```
dx/dt = −9.999 x + 9.999 y
dy/dt = 27.992 x − 0.999 y − 1.000 xz
dz/dt = −2.666 z + 1.000 xy
```

<p align="center"><img src="docs/lorenz.png" width="900" alt="Lorenz: data vs discovered model"></p>

Simulating the discovered model reproduces the attractor. Individual trajectories separate after a few Lyapunov times, as they must for any chaotic system with coefficients accurate to 10⁻⁴.

### 2. Burgers' equation (PDE-FIND, 12 candidate terms)

Data come from u<sub>t</sub> + u u<sub>x</sub> = 0.1 u<sub>xx</sub>, solved with a pseudo-spectral method on 256 points, at 101 snapshots.

| noise | discovered | error in ν | error in u u<sub>x</sub> coefficient |
|---|---|---|---|
| 0 % | u_t = 0.0996 u_xx − 0.9981 u u_x | 0.35 % | 0.19 % |
| 0.1 % | u_t = 0.1005 u_xx − 0.9992 u u_x | 0.49 % | 0.08 % |
| 0.5 % | u_t = 0.0986 u_xx − 0.9964 u u_x | 1.41 % | 0.36 % |
| 1 % | 3 spurious terms, ν off by 30 % | ✗ | – |

<p align="center"><img src="docs/burgers.png" width="820" alt="Burgers data and noise robustness"></p>

**Limitation.** The weak point of PDE-FIND is differentiating noisy data twice. Up to 0.5 % noise the local-polynomial derivatives are good enough. At 1 % the error in u<sub>xx</sub> leaks into spurious diffusive terms. The standard remedies are denoising (e.g. SVD truncation), weak/integral formulations of SINDy, and more data.

### 3. 2D Navier–Stokes: discovering vorticity transport

The data come from a pseudo-spectral simulation of decaying 2D turbulence: ω<sub>t</sub> + **u**·∇ω = ν∇²ω with ν = 0.005, on a 128² grid, with 81 snapshots. The library has **24 candidate terms**, {1, ω, u, v} × {1, ω<sub>x</sub>, ω<sub>y</sub>, ω<sub>xx</sub>, ω<sub>xy</sub>, ω<sub>yy</sub>}, sampled at 40 000 random space–time points.

<p align="center"><img src="docs/ns_coefficients.png" width="820" alt="Discovered NS coefficients"></p>

Sparse regression keeps **exactly the four true terms**. The advective coefficients come out as −0.99986 and −0.99980, and the viscosity as 0.00500 (**0.09 % error**). Nothing about the physics was built in except the choice of candidate terms.

## Usage

```bash
pip install -r requirements.txt
python run.py     # all three studies, figures and GIF (~2 min)
pytest            # spectral derivative, STLSQ, Lorenz, Burgers, vorticity equation
```

## References

S. L. Brunton, J. L. Proctor, J. N. Kutz, *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*, PNAS 113 (2016) 3932–3937.
S. H. Rudy, S. L. Brunton, J. L. Proctor, J. N. Kutz, *Data-driven discovery of partial differential equations*, Science Advances 3 (2017) e1602614.
S. L. Brunton, J. N. Kutz, *Data-Driven Science and Engineering*, 2nd ed., Cambridge University Press, 2022.

## License

MIT
