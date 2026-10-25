# [Extended Kalman filter](@id ekf-tutorial)

`ExtendedKalmanFilter` stores the mean `x` and the covariance `P`. The predict step integrates continuous dynamics and a state-transition matrix. The update uses the measurement Jacobian and the Joseph form.

The two-body demo is `demo/demo_twobody_ekf.jl`.

## Motivation

Kalman's filter is the minimum-mean-square estimator for a linear-Gaussian model. Orbit determination, and most sensor models, are nonlinear, so the extended Kalman filter linearizes the dynamics and the measurement about the current estimate and reuses that same predict/update algebra. The cost is one ODE integration plus the variational equations for the state-transition matrix, then a Joseph covariance update. Use it when Jacobians are available and the estimate stays close enough that a first-order Taylor remainder is small. It is the usual first filter on a new problem.

The linearization can underestimate covariance or diverge when the error is large, the map is strongly nonlinear, or a predict interval is long. The Joseph form ``P_{k:k} = (I-KH)P_{k:k-1}(I-KH)^\top + KRK^\top`` is used here instead of ``P_{k:k-1} - K S_{y,k:k-1} K^\top`` because it stays symmetric and is less sensitive to roundoff in ``K``. For the two-body demo the dynamics Jacobian is the state-transition matrix from those variational equations, and the range / range-rate Jacobian is taken with ForwardDiff.

**Further reading.**

- Kalman, R. E. (1960). [A New Approach to Linear Filtering and Prediction Problems](https://doi.org/10.1115/1.3662552). *Journal of Basic Engineering*, 82(1), 35–45.
- Smith, G. L., Schmidt, S. F., and McGee, L. A. (1962). [Application of Statistical Filter Theory to the Optimal Estimation of Position and Velocity on Board a Circumlunar Vehicle](https://ntrs.nasa.gov/citations/19620012057). NASA TR R-135.
- Bucy, R. S., and Joseph, P. D. (1968). *Filtering for Stochastic Processes with Applications to Guidance*. Interscience.
- Jazwinski, A. H. (1970). *Stochastic Processes and Filtering Theory*. Academic Press.
- Gelb, A., ed. (1974). *Applied Optimal Estimation*. MIT Press.
- Crassidis, J. L., and Junkins, J. L. (2012). *Optimal Estimation of Dynamic Systems*, 2nd ed. CRC Press.

## Prediction

The continuous model is ``\dot x = f_c(t, x)``. Over ``[t_{k-1}, t_k]``, `propagate!` integrates the augmented state ``[x; \mathrm{vec}(\Phi)]`` from the posterior ``x_{k-1:k-1}`` and ``\Phi(t_{k-1}) = I``:

```math
\dot\Phi = F_c(t, x)\, \Phi, \qquad
F_c = \frac{\partial f_c}{\partial x}.
```

The predicted mean ``x_{k:k-1}`` is the integrated state. The predicted covariance is

```math
P_{k:k-1} = \Phi P_{k-1:k-1} \Phi^\top + Q(\Delta t),
```

then symmetrized. For [`UnbiasedRandomProcess`](@ref) on a 6-state position/velocity model,

```math
Q(\Delta t) = \sigma_p^2
\begin{bmatrix}
(\Delta t^3/3)\, I_3 & (\Delta t^2/2)\, I_3 \\
(\Delta t^2/2)\, I_3 & \Delta t\, I_3
\end{bmatrix}.
```

Process noise is added once, at the end of the step. Along a saved trajectory, ``P(t) = \Phi(t) P_{k-1:k-1} \Phi(t)^\top`` does not yet include ``Q``.

## Update

[`NonlinearMeasurementModel`](@ref) supplies the predicted measurement ``\hat y_{k:k-1}`` and ``H`` through `get_h_H(t, x)` evaluated at the prior:

```math
\hat y_{k:k-1} = h(t_k, x_{k:k-1}), \qquad
H = \frac{\partial h}{\partial x}(t_k, x_{k:k-1}).
```

```math
S_{y,k:k-1} = H P_{k:k-1} H^\top + R, \qquad
K = P_{k:k-1} H^\top S_{y,k:k-1}^{-1},
```

```math
x_{k:k} = x_{k:k-1} + K\bigl(y_{k} - \hat y_{k:k-1}\bigr),
```

```math
P_{k:k}
  = (I - KH)\, P_{k:k-1}\, (I - KH)^\top + K R K^\top.
```

``P_{k:k}`` is symmetrized after the update.

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

After the figure, the demo prints elapsed time, process CPU time, and allocations for this filter and the square-root EKF. The two-body pair both integrate `twobody!`. The linear kernel is 5000 steps of a 6-state model with constant `F` and `H` and no ODE: the covariance column is the same Joseph update as above, with ``x_{k:k-1} = F x_{k-1:k-1}`` and ``P_{k:k-1} = F P_{k-1:k-1} F^\top + Q``, and the other column is [`SquareRootExtendedKalmanFilter`](@ref).

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
