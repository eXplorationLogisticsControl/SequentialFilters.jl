# [Square-root CKF (SRCKF)](@id srckf-tutorial)

`SquareRootCubatureKalmanFilter` is the Arasaratnam and Haykin square-root cubature Kalman filter with additive noise. The covariance is a lower-triangular factor `S` with a positive diagonal and

```math
P = S S^\top.
```

The constructor factors `P` once. Later steps do not form `P` and call `cholesky`. The element type follows `eltype(x)` and `eltype(P)`.

The point set is [`cubature_points`](@ref): `2n` points, ``\xi^{(i)} = \sqrt{n}\, e_i`` and ``\xi^{(n+i)} = -\sqrt{n}\, e_i``, each with weight ``1/(2n)``. Every weight is positive and the weights sum to one. The third-degree spherical-radial rule integrates polynomials through total degree 3 exactly, so a linear-Gaussian predict and update match the Kalman filter up to roundoff. The two-body demo is `demo/demo_srckf.jl`.

## Prediction

The filter points are

```math
\chi^{(i)} = x + S \xi^{(i)}.
```

Pass each point through the noise-free model and set

```math
\hat x = \sum_{i=1}^{2n} \frac{1}{2n}\, f(t, \chi^{(i)}, \Delta t), \qquad
d^{(i)} = f(t, \chi^{(i)}, \Delta t) - \hat x.
```

Let ``S_q`` satisfy ``Q = S_q S_q^\top``. The predicted factor is the thin QR of the columns ``\sqrt{1/(2n)}\, d^{(i)}`` together with the columns of ``S_q``. The stored factor is the sign-corrected ``R^\top``, the same QR path as a nonnegative-weight square-root CUT. No cubature weight is negative, so the predict does not downdate. The mean is ``\hat x`` and time advances by ``\Delta t``.

## Update

Redraw ``\chi^{(i)} = \hat x + S^- \xi^{(i)}`` and set ``\gamma^{(i)} = h(t, \chi^{(i)})``, ``\hat y = \sum_i w_i \gamma^{(i)}``. The innovation factor ``S_y`` is the same thin QR of the weighted measurement deviations and a factor of `R`. The cross-covariance is

```math
P_{xy} = \sum_{i=1}^{2n} \frac{1}{2n} (\chi^{(i)} - \hat x)(\gamma^{(i)} - \hat y)^\top,
\qquad
K = P_{xy} (S_y S_y^\top)^{-1}.
```

With ``U = K S_y``, the posterior factor downdates one column of `U` at a time,

```math
S^+ (S^+)^\top = S^- (S^-)^\top - \sum_{j=1}^{n_y} U_{:,j} U_{:,j}^\top,
```

and ``x \leftarrow x + K(y - \hat y)``. That is the additive update ``P - K S_y S_y^\top K^\top``. The downdate is the measurement update, not a negative cubature weight.

## Two-body demo

`demo/demo_srckf.jl` uses the EKF demo's two-body dynamics, initial condition, range and range-rate measurements, and noise. The plotted error is the estimate minus the truth. Bands are ``\pm 3\sqrt{P_{ii}}`` with ``P = SS^\top``. The figure is written to `demo/output/srckf_twobody_errors.png`.

```julia
using SequentialFilters

include("twobody_scenario.jl")
case = load_twobody_case()
filt = SquareRootCubatureKalmanFilter(0.0, case.x_hat0, case.P0)
for k in 1:case.n_meas
    predict!(filt, case.f, case.Q, case.Δt)
    update!(filt, case.h, case.ys[k], case.R)
end
```

The script prints elapsed time, process CPU time, and allocations for this filter. This package has no covariance-form cubature filter, so the demo does not print a square-root / covariance ratio. The linear kernel is 5000 steps of a 6-state model with ``f(x) = Fx`` and no ODE.

```julia
print_filter_costs(case, :srckf)
```

```bash
julia --project=demo demo/demo_srckf.jl
```

## Same calls on a linear model

```@example srckf-step
using LinearAlgebra
using SequentialFilters

F = [1.0 0.5; 0.0 1.0]
H = [1.0 0.0]
Q = [1.0e-3 0.0; 0.0 1.0e-3]
R = [0.05;;]
filt = SquareRootCubatureKalmanFilter(0.0, [0.0, 1.0], Matrix{Float64}(I, 2, 2))
predict!(filt, (t, x, Δt) -> F * x, Q, 1.0)
update!(filt, (t, x) -> H * x, [0.2], R)
(filt.x, covariance(filt))
```
