# [Square-root UKF (SRUKF)](@id srukf-tutorial)

`SquareRootUnscentedKalmanFilter` is the van der Merwe and Wan square-root unscented Kalman filter with additive noise. The covariance is a lower-triangular factor `S` with a positive diagonal and

```math
P = S S^\top.
```

Weights and ``\lambda`` are [`unscented_weights`](@ref), the same scaled convention as [`UnscentedKalmanFilter`](@ref). Defaults are `alpha = 1e-3`, `beta = 2`, and `kappa = 0`. Sigma points are columns of `S` scaled by ``\sqrt{n+\lambda}``. No Jacobians are required.

With the defaults, the center covariance weight is negative. The time update and the innovation factor put the positive weights into a thin QR, then apply a Cholesky rank-one downdate for that negative weight. The posterior factor is the Joseph QR in the update section. The two-body demo is `demo/demo_srukf.jl`.

## Motivation

The [UKF](@ref ukf-tutorial) already spends a Cholesky of ``P`` to form sigma points, then rebuilds ``P`` from a weighted outer-product sum. The default center covariance weight is negative, so that sum is not a Gram matrix: forming ``P`` and factoring it again is both wasteful and a place for roundoff to destroy definiteness. The square-root UKF keeps ``S``, draws sigma points from its columns, puts the positive-weight deviations into a thin QR with a process-noise factor, and applies a Cholesky rank-one downdate for the negative center weight.

Use it whenever you would use the UKF and want the stored uncertainty to remain a valid factor. The mean, scaled weights, and additive-noise model are those of [`UnscentedKalmanFilter`](@ref). The extra cost relative to the covariance form is the QR, the negative-weight downdate, and the Joseph posterior QR; the two-body pair still propagates the same ``2n+1`` points.

**Further reading.**

- van der Merwe, R., and Wan, E. A. (2001). [The Square-Root Unscented Kalman Filter for State and Parameter-Estimation](https://doi.org/10.1109/ICASSP.2001.940586). *Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing*, 3461–3464.
- van der Merwe, R. (2004). *Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models*. PhD dissertation, OGI School of Science & Engineering, Oregon Health & Science University.
- Zanetti, R., and DeMars, K. J. (2013). [Joseph Formulation of Unscented and Quadrature Filters with Application to Consider States](https://doi.org/10.2514/1.59935). *Journal of Guidance, Control, and Dynamics*, 36(6), 1860–1863.

The scaled weights and the time update are those of Wan and van der Merwe, cited on the [UKF page](@ref ukf-tutorial). The measurement update uses the Joseph formula of Zanetti and DeMars.

## Prediction

Sigma points use the stored posterior factor in place of a Cholesky factor of ``P_{k-1:k-1}``:

```math
\chi_{k-1:k-1}^{(0)} = x_{k-1:k-1}, \qquad
\chi_{k-1:k-1}^{(i)} = x_{k-1:k-1} + \sqrt{c}\, \bigl(S_{k-1:k-1}\bigr)_{:,i},
```

```math
\chi_{k-1:k-1}^{(n+i)} = x_{k-1:k-1} - \sqrt{c}\, \bigl(S_{k-1:k-1}\bigr)_{:,i},
```

with ``c = n + \lambda``. Propagate ``\chi_{k:k-1}^{(i)} = f\bigl(t_{k-1}, \chi_{k-1:k-1}^{(i)}, \Delta t\bigr)`` and

```math
x_{k:k-1} = \sum_{i=0}^{2n} W_m^{(i)}\, \chi_{k:k-1}^{(i)}, \qquad
d^{(i)} = \chi_{k:k-1}^{(i)} - x_{k:k-1}.
```

Let ``S_q`` satisfy ``Q = S_q S_q^\top``. Columns ``\sqrt{W_c^{(i)}}\, d^{(i)}`` with ``W_c^{(i)} > 0``, and the columns of ``S_q``, form the matrix whose thin QR becomes a lower factor ``S_{\mathrm{qr}}`` by the same sign-flip rule as the [Square-root EKF (SREKF)](@ref srekf-tutorial). Outer weights ``i \ge 1`` are required to be positive.

If ``W_c^{(0)} > 0``, the center column is included in that QR. If ``W_c^{(0)} < 0``, it is a rank-one downdate:

```math
S_{k:k-1} S_{k:k-1}^\top
  = S_{\mathrm{qr}} S_{\mathrm{qr}}^\top
  - \big(\sqrt{-W_c^{(0)}}\, d^{(0)}\big)
    \big(\sqrt{-W_c^{(0)}}\, d^{(0)}\big)^\top.
```

For a lower factor `L` and a vector `w`, one rank-one step with sign ``\sigma = \pm 1`` walks ``\ell = 1,\ldots,n``:

```math
r = \sqrt{L_{\ell\ell}^2 + \sigma w_\ell^2}, \quad
c_\ell = r / L_{\ell\ell}, \quad
s_\ell = w_\ell / L_{\ell\ell}, \quad
L_{\ell\ell} \leftarrow r,
```

```math
L_{j\ell} \leftarrow (L_{j\ell} + \sigma s_\ell w_j) / c_\ell, \qquad
w_j \leftarrow c_\ell w_j - s_\ell L_{j\ell}^{\mathrm{new}}, \qquad j > \ell.
```

``\sigma = -1`` is the downdate. The diagonal stays positive, or the step errors if the downdate would lose definiteness. The mean is ``x_{k:k-1}`` and time advances by ``\Delta t``.

## Update

Redraw sigma points from ``\bigl(x_{k:k-1}, S_{k:k-1}\bigr)`` and set ``\gamma^{(i)} = h\bigl(t_k, \chi_{k:k-1}^{(i)}\bigr)``, ``\hat y_{k:k-1} = \sum W_m^{(i)} \gamma^{(i)}``. The innovation factor ``S_{y,k:k-1}`` is the same QR-and-downdate map applied to the measurement deviations and a factor of `R`:

```math
S_{y,k:k-1} S_{y,k:k-1}^\top
  = \sum_{i=0}^{2n} W_c^{(i)}
    \bigl(\gamma^{(i)} - \hat y_{k:k-1}\bigr)
    \bigl(\gamma^{(i)} - \hat y_{k:k-1}\bigr)^\top + R.
```

The cross-covariance is the weighted outer product used by the covariance-form UKF,

```math
P_{xy,k:k-1} = \sum_{i=0}^{2n} W_c^{(i)}
    \bigl(\chi_{k:k-1}^{(i)} - x_{k:k-1}\bigr)
    \bigl(\gamma^{(i)} - \hat y_{k:k-1}\bigr)^\top,
```

```math
K = P_{xy,k:k-1}
    \bigl(S_{y,k:k-1} S_{y,k:k-1}^\top\bigr)^{-1}.
```

The posterior factor is the Joseph form, the same thin QR as the [Square-root EKF (SREKF)](@ref srekf-tutorial), with the cross-covariance in place of ``H S``. Define

```math
C = \bigl(S_{k:k-1}^{-1} P_{xy,k:k-1}\bigr)^\top,
```

so ``S_{k:k-1} C^\top = P_{xy,k:k-1}``. Let ``S_r`` be a factor of the residual measurement covariance,

```math
S_r S_r^\top = S_{y,k:k-1} S_{y,k:k-1}^\top - C C^\top.
```

For a linear measurement, ``C = H S_{k:k-1}`` and that residual is ``R``. The stacked matrix

```math
\begin{bmatrix}
\bigl(S_{k:k-1} - K C\bigr)^\top \\
(K S_r)^\top
\end{bmatrix}
```

has thin QR equal to a lower factor ``S_{k:k}`` with a positive diagonal, so

```math
S_{k:k} S_{k:k}^\top
  = \bigl(S_{k:k-1} - K C\bigr)\bigl(S_{k:k-1} - K C\bigr)^\top
    + K S_r S_r^\top K^\top,
```

which is the Zanetti–DeMars formula

```math
P_{k:k}
  = P_{k:k-1}
  - P_{xy,k:k-1} K^\top
  - K P_{xy,k:k-1}^\top
  + K S_{y,k:k-1} S_{y,k:k-1}^\top K^\top.
```

The mean update is unchanged,

```math
x_{k:k} = x_{k:k-1} + K\bigl(y_{k} - \hat y_{k:k-1}\bigr).
```

For the optimal gain this matches ``P_{k:k-1} - K S_{y,k:k-1} S_{y,k:k-1}^\top K^\top`` in exact arithmetic. The Joseph QR stays valid when ``K`` is not that gain.

## Two-body demo

`demo/demo_srukf.jl` uses the EKF demo's two-body dynamics, initial condition, range and range-rate measurements, and noise. The plotted error is the estimate minus the truth. Bands are ``\pm 3\sqrt{P_{ii}}`` with ``P = SS^\top``.

```julia
using SequentialFilters

include("twobody_scenario.jl")
case = load_twobody_case()
filt = SquareRootUnscentedKalmanFilter(0.0, case.x_hat0, case.P0)
for k in 1:case.n_meas
    predict!(filt, case.f, case.Q, case.Δt)
    update!(filt, case.h, case.ys[k], case.R)
end
```

The script prints elapsed time, process CPU time, and allocations for `UnscentedKalmanFilter` and this filter. The two-body pair propagates the same sigma points. The linear kernel is 5000 steps of a 6-state model with ``f(x) = Fx`` and no ODE, which is where the QR and downdate cost shows up next to the covariance-form sums.

```julia
print_filter_costs(case, :ukf)
```

```bash
julia --project=demo demo/demo_srukf.jl
```

## Same calls on a linear model

```@example srukf-step
using LinearAlgebra
using SequentialFilters

F = [1.0 0.5; 0.0 1.0]
H = [1.0 0.0]
Q = [1.0e-3 0.0; 0.0 1.0e-3]
R = [0.05;;]
filt = SquareRootUnscentedKalmanFilter(0.0, [0.0, 1.0], Matrix{Float64}(I, 2, 2))
predict!(filt, (t, x, Δt) -> F * x, Q, 1.0)
update!(filt, (t, x) -> H * x, [0.2], R)
(filt.x, covariance(filt))
```
