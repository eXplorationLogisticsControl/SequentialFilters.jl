# [Square-root EKF](@id srekf-tutorial)

`SquareRootExtendedKalmanFilter` is a discrete extended Kalman filter. It stores the covariance as a lower-triangular factor `S` with a positive diagonal and

```math
P = S S^\top.
```

The constructor factors `P` once. Later predict and update steps rebuild `S` with a thin QR. They do not form `P` and call `cholesky` again. `covariance(filter)` returns the symmetrized product ``SS^\top``.

`f` and `h` are nonlinear. Their Jacobians are an `nx × nx` or `ny × nx` matrix, or a function of `(t, x, Δt)` and `(t, x)` respectively. Process and measurement noise are additive. The two-body demo is `demo/demo_srekf.jl`. There `F` is the state-transition matrix from the same propagation the EKF uses.

## Prediction

```math
x^- = f(t, x, \Delta t), \qquad
F = \frac{\partial f}{\partial x}(t, x, \Delta t).
```

Let ``S_q`` be a factor of the process noise, ``Q = S_q S_q^\top`` (lower Cholesky when `Q` is positive definite, and an eigenvalue factor when it is only semidefinite). The predicted factor is the thin QR of the stacked transpose

```math
\begin{bmatrix} (F S)^\top \\ S_q^\top \end{bmatrix}
= Q_r R.
```

Rows of ``R`` whose diagonal entry is negative are multiplied by ``-1``, and the stored factor is the lower triangle of ``R^\top``. That gives

```math
S^- (S^-)^\top = F S S^\top F^\top + Q
```

with a nonnegative diagonal. The mean is ``x^-`` and time advances by ``\Delta t``.

## Update

```math
\hat y = h(t, x), \qquad H = \frac{\partial h}{\partial x}(t, x),
```

and ``R = S_r S_r^\top``. The innovation factor is the same QR map applied to ``[(HS)^\top;\; S_r^\top]``:

```math
S_y S_y^\top = H S S^\top H^\top + R.
```

The gain uses that factor rather than a fresh covariance inverse,

```math
K = S (HS)^\top (S_y S_y^\top)^{-1}.
```

The posterior factor is the Joseph form, again as a thin QR,

```math
\begin{bmatrix} ((I - KH) S)^\top \\ (K S_r)^\top \end{bmatrix},
```

so

```math
S^+ (S^+)^\top
  = (I - KH)\, S S^\top\, (I - KH)^\top + K R K^\top,
```

and ``x \leftarrow x + K(y - \hat y)``.

`update!(filter, model::MeasurementModel, y, R)` reads ``(\hat y, H)`` from `get_h_H`, which is the same measurement object the EKF update accepts.

## Two-body demo

`demo/demo_srekf.jl` loads the EKF demo's orbit, initial condition, measurements, `R`, and `Q`. The error plotted is the estimate minus the truth. Bands are ``\pm 3\sqrt{P_{ii}}`` with ``P = SS^\top``.

```julia
using SequentialFilters

include("twobody_scenario.jl")
case = load_twobody_case()
filt = SquareRootExtendedKalmanFilter(0.0, case.x_hat0, case.P0)
for k in 1:case.n_meas
    predict!(filt, case.f, case.F, case.Q, case.Δt)
    update!(filt, case.h, case.H, case.ys[k], case.R)
end
```

`case.f` and `case.F` share one ODE propagation per state: `f` returns the propagated state and `F` returns ``\Phi``. The script then prints elapsed time, process CPU time, and allocations against `ExtendedKalmanFilter` on that same orbit, and against the Joseph covariance update on a 5000-step linear kernel with no ODE.

```julia
print_filter_costs(case, :ekf)
```

```bash
julia --project=demo demo/demo_srekf.jl
```

## Same calls on a linear model

```@example srekf-step
using LinearAlgebra
using SequentialFilters

F = [1.0 0.5; 0.0 1.0]
H = [1.0 0.0]
Q = [1.0e-3 0.0; 0.0 1.0e-3]
R = [0.05;;]
filt = SquareRootExtendedKalmanFilter(0.0, [0.0, 1.0], Matrix{Float64}(I, 2, 2))
predict!(filt, (t, x, Δt) -> F * x, F, Q, 1.0)
update!(filt, (t, x) -> H * x, H, [0.2], R)
(filt.x, covariance(filt))
```
