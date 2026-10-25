# [Extended Kalman filter](@id ekf-tutorial)

`ExtendedKalmanFilter` stores the mean `x` and the covariance `P`. The predict step integrates continuous dynamics and a state-transition matrix. The update uses the measurement Jacobian and the Joseph form.

The two-body demo is `demo/demo_twobody_ekf.jl`.

## Prediction

The continuous model is ``\dot x = f_c(t, x)``. Over ``[t_k, t_k+\Delta t]``, `propagate!` integrates the augmented state ``[x; \mathrm{vec}(\Phi)]`` from ``\Phi(t_k) = I``:

```math
\dot\Phi = F_c(t, x)\, \Phi, \qquad
F_c = \frac{\partial f_c}{\partial x}.
```

The predicted mean is the integrated state. The predicted covariance is

```math
P_{k+1|k} = \Phi P_{k|k} \Phi^\top + Q(\Delta t),
```

then symmetrized. For [`UnbiasedRandomProcess`](@ref) on a 6-state position/velocity model,

```math
Q(\Delta t) = \sigma_p^2
\begin{bmatrix}
(\Delta t^3/3)\, I_3 & (\Delta t^2/2)\, I_3 \\
(\Delta t^2/2)\, I_3 & \Delta t\, I_3
\end{bmatrix}.
```

Process noise is added once, at the end of the step. Along a saved trajectory, ``P(t) = \Phi(t) P_{k|k} \Phi(t)^\top`` does not yet include ``Q``.

## Update

[`NonlinearMeasurementModel`](@ref) supplies ``\hat y`` and ``H`` through `get_h_H(t, x)`:

```math
\hat y = h(t, x), \qquad H = \frac{\partial h}{\partial x}.
```

```math
S = H P H^\top + R, \qquad K = P H^\top S^{-1},
```

```math
x \leftarrow x + K(y - \hat y),
```

```math
P \leftarrow (I - KH)\, P\, (I - KH)^\top + K R K^\top.
```

`P` is symmetrized after the update.

## Two-body demo

The script below is the filter loop from `demo/demo_twobody_ekf.jl`. Dynamics, initial conditions, range and range-rate measurements, and the noise covariances are the shared two-body case. The figure records the estimation error, which is the estimate minus the truth, together with ``\pm 3\sqrt{P_{ii}}``.

```julia
using SequentialFilters

ekf = ExtendedKalmanFilter(0.0, x_hat0, P0)
dynamics = NonlinearODEModel(twobody!; n = n, p = μ)
process_noise_model = UnbiasedRandomProcess(1.0e-6)
measurements = NonlinearMeasurementModel(get_h_H)

for k in 1:n_meas
    predict!(ekf, dynamics, process_noise_model, Δt)
    update!(ekf, measurements, ys[k], R)
end
```

`get_h_H(t, x)` returns `(range_rangerate(t, x), H)` with `H` from `ForwardDiff`. The full script also samples the predicted trajectory between measurements and writes `demo/twobody_ekf_errors.png`.

After the figure, the demo prints elapsed time, process CPU time, and allocations for this filter and the square-root EKF. The two-body pair both integrate `twobody!`. The linear kernel is 5000 steps of a 6-state model with constant `F` and `H` and no ODE: the covariance column is the same Joseph update as above, with ``x \leftarrow Fx`` and ``P \leftarrow FPF^\top + Q``, and the other column is [`SquareRootExtendedKalmanFilter`](@ref).

```julia
include(joinpath(@__DIR__, "benchmark_filters.jl"))
print_filter_costs(load_twobody_case(), :ekf)
```

Run it with

```bash
julia --project=demo demo/demo_twobody_ekf.jl
```

## Same calls on a coasting state

`UnbiasedRandomProcess` is the 6-state noise model used by the two-body demo, so this check uses that dimension. One predict and one range update:

```@example ekf-step
using LinearAlgebra
using SequentialFilters

function coast!(du, u, p, t)
    du[1] = u[4]
    du[2] = u[5]
    du[3] = u[6]
    du[4] = 0.0
    du[5] = 0.0
    du[6] = 0.0
    return nothing
end

function get_h_H(t, x)
    yhat = [hypot(x[1], x[2], x[3])]
    r = yhat[1]
    H = zeros(1, 6)
    H[1, 1] = x[1] / r
    H[1, 2] = x[2] / r
    H[1, 3] = x[3] / r
    return yhat, H
end

x0 = [1.0, 0.0, 0.0, 0.1, 0.0, 0.0]
P0 = Matrix(Diagonal([1.0, 1.0, 1.0, 0.1, 0.1, 0.1] .^ 2))
ekf = ExtendedKalmanFilter(0.0, x0, P0)
predict!(ekf, NonlinearODEModel(coast!; n = 6), UnbiasedRandomProcess(1.0e-3), 1.0)
update!(ekf, NonlinearMeasurementModel(get_h_H), [1.05], [0.01;;])
ekf.x
```
