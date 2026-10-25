# [Square-root CUT (SRCUT)](@id srcut-tutorial)

`SquareRootConjugateUnscentedKalmanFilter` is the conjugate unscented Kalman filter with the covariance stored as a lower-triangular factor `S`,

```math
P = S S^\top,
```

and a positive diagonal. This is the same factor convention as the square-root UKF. The constructor factors `P` once. Later steps do not form `P` and call `cholesky`.

Sigma points and weights come from [`cut_points`](@ref). The default `order` is `4`. Supported orders are CUT4 for every state dimension `n ≥ 1`, and CUT6 for `2 ≤ n ≤ 9`. CUT uses one weight per point for both the mean and the covariance. The two-body demo is `demo/demo_srcut.jl` and uses order 4.

## Motivation

The [covariance-form CUT](@ref cut-tutorial) matches higher Gaussian moments than the UKF, then stores ``P`` and adds the weighted outer products directly. That is the same numerical exposure as the covariance-form UKF: after enough steps, or when a weight is negative, the reconstructed ``P`` can lose definiteness, and the next predict has to factor it again. The square-root CUT keeps a factor ``S``. Predict and the innovation factor apply the van der Merwe / Wan QR-and-downdate to the CUT weights; the posterior is the Joseph QR.

Use it when you want CUT moment matching and a stored factor that stays positive definite. CUT4 and CUT6 weights on the supported dimensions are nonnegative, so a predict is usually only the QR. There is no separate "square-root CUT" paper: the cubature is Adurthi, Singla, and Singh. The predict and the innovation factor use the van der Merwe / Wan QR-and-downdate. The posterior is the Joseph QR of the [square-root UKF](@ref srukf-tutorial).

**Further reading.**

- Adurthi, N., Singla, P., and Singh, T. (2018). [Conjugate Unscented Transformation: Applications to Estimation and Control](https://doi.org/10.1115/1.4037783). *ASME Journal of Dynamic Systems, Measurement, and Control*, 140(3), 030907.
- van der Merwe, R., and Wan, E. A. (2001). [The Square-Root Unscented Kalman Filter for State and Parameter-Estimation](https://doi.org/10.1109/ICASSP.2001.940586). *Proc. IEEE ICASSP*, 3461–3464.
- Zanetti, R., and DeMars, K. J. (2013). [Joseph Formulation of Unscented and Quadrature Filters with Application to Consider States](https://doi.org/10.2514/1.59935). *Journal of Guidance, Control, and Dynamics*, 36(6), 1860–1863.

The CUT orders are those of the [CUT](@ref cut-tutorial) tutorial. The predict QR is written below. The measurement update is the Joseph QR of the [square-root UKF](@ref srukf-tutorial).

## Prediction

Let ``(\xi^{(i)}, w_i)`` be the standard-normal CUT points. The filter points are

```math
\chi_{k-1:k-1}^{(i)} = x_{k-1:k-1} + S_{k-1:k-1} \xi^{(i)}.
```

Pass each point through the noise-free model and set

```math
x_{k:k-1} = \sum_i w_i\, f\bigl(t_{k-1}, \chi_{k-1:k-1}^{(i)}, \Delta t\bigr),
```

```math
d^{(i)} = f\bigl(t_{k-1}, \chi_{k-1:k-1}^{(i)}, \Delta t\bigr) - x_{k:k-1}.
```

Let ``S_q`` satisfy ``Q = S_q S_q^\top``. Every point with ``w_i > 0`` contributes the column ``\sqrt{w_i}\, d^{(i)}``. Those columns and the columns of ``S_q`` form the thin QR whose lower factor ``S_{\mathrm{qr}}`` is the sign-corrected ``R^\top``, as in the square-root EKF. A zero weight is left out.

A negative weight is not put in the QR. After ``S_{\mathrm{qr}}`` is built, each negative weight is a Cholesky rank-one downdate in column order. For a lower factor `L` and vector ``\sqrt{-w_i}\, d^{(i)}``, the ``\sigma = -1`` step is the same recurrence as the square-root UKF:

```math
r = \sqrt{L_{\ell\ell}^2 - w_\ell^2}, \quad
c_\ell = r / L_{\ell\ell}, \quad
s_\ell = w_\ell / L_{\ell\ell}, \quad
L_{\ell\ell} \leftarrow r,
```

```math
L_{j\ell} \leftarrow (L_{j\ell} - s_\ell w_j) / c_\ell, \qquad
w_j \leftarrow c_\ell w_j - s_\ell L_{j\ell}^{\mathrm{new}}, \qquad j > \ell.
```

Here ``w`` in the recurrence is the downdate vector, not the CUT weight. The result satisfies

```math
S_{k:k-1} S_{k:k-1}^\top
  = \sum_{w_i > 0} w_i\, d^{(i)} (d^{(i)})^\top
  + Q
  - \sum_{w_i < 0} |w_i|\, d^{(i)} (d^{(i)})^\top,
```

which is the CUT covariance ``P_{k:k-1} = \sum_i w_i d^{(i)}(d^{(i)})^\top + Q``. CUT4 and CUT6 weights on the supported dimensions are nonnegative, so those predicts only use the QR. The downdate is how a negative CUT weight would be applied to ``S_{k:k-1}``. The mean is ``x_{k:k-1}`` and time advances by ``\Delta t``.

## Update

Redraw ``\chi_{k:k-1}^{(i)} = x_{k:k-1} + S_{k:k-1} \xi^{(i)}`` and set ``\gamma^{(i)} = h\bigl(t_k, \chi_{k:k-1}^{(i)}\bigr)``, ``\hat y_{k:k-1} = \sum_i w_i \gamma^{(i)}``. The innovation factor ``S_{y,k:k-1}`` is the same QR and negative-weight downdate applied to the measurement deviations and a factor of `R`. The cross-covariance keeps the CUT weights, including any negative ones:

```math
P_{xy,k:k-1}
  = \sum_i w_i
    \bigl(\chi_{k:k-1}^{(i)} - x_{k:k-1}\bigr)
    \bigl(\gamma^{(i)} - \hat y_{k:k-1}\bigr)^\top,
```

```math
K = P_{xy,k:k-1}
    \bigl(S_{y,k:k-1} S_{y,k:k-1}^\top\bigr)^{-1}.
```

The posterior factor is the Joseph form, the same thin QR as the [square-root UKF](@ref srukf-tutorial), with the cross-covariance in place of ``H S``. Define

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

`demo/demo_srcut.jl` uses the EKF demo's two-body dynamics, initial condition, range and range-rate measurements, and noise. The filter order is 4. The plotted error is the estimate minus the truth. Bands are ``\pm 3\sqrt{P_{ii}}`` with ``P = SS^\top``. The figure is written to `demo/output/srcut_twobody_errors.png`.

```julia
using SequentialFilters

include("twobody_scenario.jl")
case = load_twobody_case()
filt = SquareRootConjugateUnscentedKalmanFilter(0.0, case.x_hat0, case.P0; order = 4)
for k in 1:case.n_meas
    predict!(filt, case.f, case.Q, case.Δt)
    update!(filt, case.h, case.ys[k], case.R)
end
```

The script then prints elapsed time, process CPU time, and allocations for `ConjugateUnscentedKalmanFilter` and this filter at order 4. Both two-body runs propagate the same CUT points. The linear kernel is 5000 steps of a 6-state model with ``f(x) = Fx`` and no ODE.

```julia
print_filter_costs(case, :cut)
```

```bash
julia --project=demo demo/demo_srcut.jl
```

## Same calls on a linear model

```@example srcut-step
using LinearAlgebra
using SequentialFilters

F = [1.0 0.5; 0.0 1.0]
H = [1.0 0.0]
Q = [1.0e-3 0.0; 0.0 1.0e-3]
R = [0.05;;]
filt = SquareRootConjugateUnscentedKalmanFilter(0.0, [0.0, 1.0], Matrix{Float64}(I, 2, 2); order = 4)
predict!(filt, (t, x, Δt) -> F * x, Q, 1.0)
update!(filt, (t, x) -> H * x, [0.2], R)
(filt.x, covariance(filt))
```
