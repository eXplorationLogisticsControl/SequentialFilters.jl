# [Square-root CUT](@id srcut-tutorial)

`SquareRootConjugateUnscentedKalmanFilter` is the conjugate unscented Kalman filter with the covariance stored as a lower-triangular factor `S`,

```math
P = S S^\top,
```

and a positive diagonal. This is the same factor convention as the square-root UKF. The constructor factors `P` once. Later steps do not form `P` and call `cholesky`.

Sigma points and weights come from [`cut_points`](@ref). The default `order` is `4`. Supported orders are CUT4 for every state dimension `n ≥ 1`, and CUT6 for `2 ≤ n ≤ 9`. CUT uses one weight per point for both the mean and the covariance. The two-body demo is `demo/demo_srcut.jl` and uses order 4.

## Prediction

Let ``(\xi^{(i)}, w_i)`` be the standard-normal CUT points. The filter points are

```math
\chi^{(i)} = x + S \xi^{(i)}.
```

Pass each point through the noise-free model and set

```math
\hat x = \sum_i w_i\, f(t, \chi^{(i)}, \Delta t), \qquad
d^{(i)} = f(t, \chi^{(i)}, \Delta t) - \hat x.
```

Let ``S_q`` satisfy ``Q = S_q S_q^\top``. Every point with ``w_i > 0`` contributes the column ``\sqrt{w_i}\, d^{(i)}``. Those columns and the columns of ``S_q`` form the thin QR whose lower factor ``S_{\mathrm{qr}}`` is the sign-corrected ``R^\top``, as in the square-root EKF. A zero weight is left out.

A negative weight is not put in the QR. After ``S_{\mathrm{qr}}`` is built, each negative weight is a Cholesky rank-one downdate in column order. For a lower factor `L` and vector ``\sqrt{-w_i}\, d^{(i)}``, the ``\sigma = -1`` step is the same recurrence as the square-root UKF:

```math
r = \sqrt{L_{kk}^2 - w_k^2}, \quad
c_k = r / L_{kk}, \quad
s_k = w_k / L_{kk}, \quad
L_{kk} \leftarrow r,
```

```math
L_{jk} \leftarrow (L_{jk} - s_k w_j) / c_k, \qquad
w_j \leftarrow c_k w_j - s_k L_{jk}^{\mathrm{new}}, \qquad j > k.
```

Here ``w`` in the recurrence is the downdate vector, not the CUT weight. The result satisfies

```math
S^- (S^-)^\top
  = \sum_{w_i > 0} w_i\, d^{(i)} (d^{(i)})^\top
  + Q
  - \sum_{w_i < 0} |w_i|\, d^{(i)} (d^{(i)})^\top,
```

which is the CUT covariance ``\sum_i w_i d^{(i)}(d^{(i)})^\top + Q``. CUT4 and CUT6 weights on the supported dimensions are nonnegative, so those predicts only use the QR. The downdate is how a negative CUT weight would be applied to `S`. The mean is ``\hat x`` and time advances by ``\Delta t``.

## Update

Redraw ``\chi^{(i)} = \hat x + S^- \xi^{(i)}`` and set ``\gamma^{(i)} = h(t, \chi^{(i)})``, ``\hat y = \sum_i w_i \gamma^{(i)}``. The innovation factor ``S_y`` is the same QR and negative-weight downdate applied to the measurement deviations and a factor of `R`. The cross-covariance keeps the CUT weights, including any negative ones:

```math
P_{xy} = \sum_i w_i (\chi^{(i)} - \hat x)(\gamma^{(i)} - \hat y)^\top,
\qquad
K = P_{xy} (S_y S_y^\top)^{-1}.
```

With ``U = K S_y``, the posterior factor downdates one column of `U` at a time,

```math
S^+ (S^+)^\top = S^- (S^-)^\top - \sum_{j=1}^{n_y} U_{:,j} U_{:,j}^\top,
```

and ``x \leftarrow x + K(y - \hat y)``. That is the additive update ``P - K S_y S_y^\top K^\top``.

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
