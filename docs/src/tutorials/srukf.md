# [Square-root UKF](@id srukf-tutorial)

`SquareRootUnscentedKalmanFilter` is the van der Merwe and Wan square-root unscented Kalman filter with additive noise. The covariance is a lower-triangular factor `S` with a positive diagonal and

```math
P = S S^\top.
```

Weights and ``\lambda`` are [`unscented_weights`](@ref), the same scaled convention as [`UnscentedKalmanFilter`](@ref). Defaults are `alpha = 1e-3`, `beta = 2`, and `kappa = 0`. Sigma points are columns of `S` scaled by ``\sqrt{n+\lambda}``. No Jacobians are required.

With the defaults, the center covariance weight is negative. The time and measurement updates put the positive weights into a thin QR, then apply a Cholesky rank-one downdate for that negative weight. The two-body demo is `demo/demo_srukf.jl`.

## Prediction

Sigma points use the stored factor in place of a Cholesky factor of `P`:

```math
\chi^{(0)} = x, \qquad
\chi^{(i)} = x + \sqrt{c}\, S_{:,i}, \qquad
\chi^{(n+i)} = x - \sqrt{c}\, S_{:,i},
```

with ``c = n + \lambda``. Propagate ``\chi^{-(i)} = f(t, \chi^{(i)}, \Delta t)`` and

```math
\hat x = \sum_{i=0}^{2n} W_m^{(i)}\, \chi^{-(i)}, \qquad
d^{(i)} = \chi^{-(i)} - \hat x.
```

Let ``S_q`` satisfy ``Q = S_q S_q^\top``. Columns ``\sqrt{W_c^{(i)}}\, d^{(i)}`` with ``W_c^{(i)} > 0``, and the columns of ``S_q``, form the matrix whose thin QR becomes a lower factor ``S_{\mathrm{qr}}`` by the same sign-flip rule as the [Square-root EKF](@ref srekf-tutorial). Outer weights ``i \ge 1`` are required to be positive.

If ``W_c^{(0)} > 0``, the center column is included in that QR. If ``W_c^{(0)} < 0``, it is a rank-one downdate:

```math
S^- (S^-)^\top
  = S_{\mathrm{qr}} S_{\mathrm{qr}}^\top
  - \big(\sqrt{-W_c^{(0)}}\, d^{(0)}\big)
    \big(\sqrt{-W_c^{(0)}}\, d^{(0)}\big)^\top.
```

For a lower factor `L` and a vector `w`, one rank-one step with sign ``\sigma = \pm 1`` walks ``k = 1,\ldots,n``:

```math
r = \sqrt{L_{kk}^2 + \sigma w_k^2}, \quad
c_k = r / L_{kk}, \quad
s_k = w_k / L_{kk}, \quad
L_{kk} \leftarrow r,
```

```math
L_{jk} \leftarrow (L_{jk} + \sigma s_k w_j) / c_k, \qquad
w_j \leftarrow c_k w_j - s_k L_{jk}^{\mathrm{new}}, \qquad j > k.
```

``\sigma = -1`` is the downdate. The diagonal stays positive, or the step errors if the downdate would lose definiteness. The mean is ``\hat x`` and time advances by ``\Delta t``.

## Update

Redraw sigma points from ``(\hat x, S^-)`` and set ``\gamma^{(i)} = h(t, \chi^{(i)})``, ``\hat y = \sum W_m^{(i)} \gamma^{(i)}``. The innovation factor ``S_y`` is the same QR-and-downdate map applied to the measurement deviations and a factor of `R`:

```math
S_y S_y^\top
  = \sum_{i=0}^{2n} W_c^{(i)} (\gamma^{(i)} - \hat y)(\gamma^{(i)} - \hat y)^\top + R.
```

The cross-covariance is the weighted outer product used by the covariance-form UKF,

```math
P_{xy} = \sum_{i=0}^{2n} W_c^{(i)} (\chi^{(i)} - \hat x)(\gamma^{(i)} - \hat y)^\top,
\qquad
K = P_{xy} (S_y S_y^\top)^{-1}.
```

With ``U = K S_y``, the posterior factor downdates one column of `U` at a time:

```math
S^+ (S^+)^\top = S^- (S^-)^\top - \sum_{j=1}^{n_y} U_{:,j} U_{:,j}^\top,
```

and ``x \leftarrow x + K(y - \hat y)``. This is ``P - K S_y S_y^\top K^\top``, the additive square-root measurement update.

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
