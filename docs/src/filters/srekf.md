# [Square-root EKF (SREKF)](@id srekf-tutorial)

`SquareRootExtendedKalmanFilter` is a discrete extended Kalman filter. It stores the covariance as a lower-triangular factor `S` with a positive diagonal and

```math
P = S S^\top.
```

The constructor factors `P` once. Later predict and update steps rebuild `S` with a thin QR. They do not form `P` and call `cholesky` again. `covariance(filter)` returns the symmetrized product ``SS^\top``.

`h` is nonlinear. Its Jacobian is an `ny × nx` matrix or a function of `(t, x)`. Process and measurement noise are additive. The two-body demo is `demo/demo_srekf.jl`. On a [`NonlinearODEModel`](@ref), `F` is the state-transition matrix from the same propagation the EKF uses. This filter follows one trajectory, so `predict!` has no `ensemblealg` keyword.

## Motivation

The [EKF](@ref ekf-tutorial) stores ``P`` and applies rank-two updates (``P_{k:k-1} = F P_{k-1:k-1} F^\top + Q``, Joseph). In finite precision those products can lose symmetry and, after enough steps or when ``R`` is small, positive-definiteness. A square-root filter stores a factor ``S`` with ``P = SS^\top``, so the reconstructed covariance cannot become indefinite. Predict and update rebuild ``S`` with a thin QR; they do not form ``P`` and call `cholesky` again.

Use this form when the estimate must stay numerically well-conditioned, or when you want the EKF's mean and Jacobians without refactoring ``P`` at every step. The mean recursion is the discrete EKF. Only the covariance arithmetic changes. `covariance(filter)` rebuilds ``P = SS^\top`` when a full matrix is needed, for example the ``\pm 3\sqrt{P_{ii}}`` bands in the demo. On a dense, low-dimensional problem the QR is extra work relative to forming ``P``; the return is the stored factor.

**Further reading.**

- Potter, J. E., and Stern, R. G. (1963). Statistical Filtering of Space Navigation Measurements. *Proc. AIAA Guidance and Control Conference*.
- Kaminski, P. G., Bryson, A. E., and Schmidt, S. F. (1971). [Discrete Square Root Filtering: A Survey of Current Techniques](https://doi.org/10.1109/TAC.1971.1099816). *IEEE Transactions on Automatic Control*, 16(6), 727–736.
- Bierman, G. J. (1977). *Factorization Methods for Discrete Sequential Estimation*. Academic Press.
- Grewal, M. S., and Andrews, A. P. (2015). *Kalman Filtering: Theory and Practice Using MATLAB*, 4th ed. Wiley.

## Prediction

```math
x_{k:k-1} = f\bigl(t_{k-1}, x_{k-1:k-1}, \Delta t\bigr), \qquad
F = \frac{\partial f}{\partial x}\bigl(t_{k-1}, x_{k-1:k-1}, \Delta t\bigr).
```

Let ``S_q`` be a factor of the process noise, ``Q = S_q S_q^\top`` (lower Cholesky when `Q` is positive definite, and an eigenvalue factor when it is only semidefinite). The predicted factor is the thin QR of the stacked transpose

```math
\begin{bmatrix} (F S_{k-1:k-1})^\top \\ S_q^\top \end{bmatrix}
= Q_r R.
```

Rows of ``R`` whose diagonal entry is negative are multiplied by ``-1``, and the stored factor is the lower triangle of ``R^\top``. That gives

```math
S_{k:k-1} S_{k:k-1}^\top
  = F S_{k-1:k-1} S_{k-1:k-1}^\top F^\top + Q
```

with a nonnegative diagonal. The mean is ``x_{k:k-1}`` and time advances by ``\Delta t``.

`predict!` takes a [`DynamicsModel`](@ref) and a [`ProcessNoiseModel`](@ref), with ``Q = `` `get_noise(process_noise_model, Δt)`. A [`NonlinearODEModel`](@ref) integrates the augmented state and returns that `ODESolution`; ``F`` is read from it. A [`DiscreteDynamicsModel`](@ref) evaluates `f(t, x, Δt)` and `discrete_jacobian` (the stored Jacobian, or ForwardDiff). The returned solution stores `[x; vec(Φ)]` on either path. There is no `ensemblealg` argument.

## Update

```math
\hat y_{k:k-1} = h\bigl(t_k, x_{k:k-1}\bigr), \qquad
H = \frac{\partial h}{\partial x}\bigl(t_k, x_{k:k-1}\bigr),
```

and ``R = S_r S_r^\top``. The innovation factor is the same QR map applied to ``[(H S_{k:k-1})^\top;\; S_r^\top]``:

```math
S_{y,k:k-1} S_{y,k:k-1}^\top
  = H S_{k:k-1} S_{k:k-1}^\top H^\top + R.
```

The gain uses that factor rather than a fresh covariance inverse,

```math
K = S_{k:k-1} (H S_{k:k-1})^\top
    \bigl(S_{y,k:k-1} S_{y,k:k-1}^\top\bigr)^{-1}.
```

The posterior factor is the Joseph form, again as a thin QR,

```math
\begin{bmatrix} ((I - KH) S_{k:k-1})^\top \\ (K S_r)^\top \end{bmatrix},
```

so

```math
S_{k:k} S_{k:k}^\top
  = (I - KH)\, S_{k:k-1} S_{k:k-1}^\top\, (I - KH)^\top + K R K^\top,
```

and

```math
x_{k:k} = x_{k:k-1} + K\bigl(y_{k} - \hat y_{k:k-1}\bigr).
```

`update!(filter, model::MeasurementModel, y, R)` reads ``(\hat y_{k:k-1}, H)`` from `get_h_H`, which is the same measurement object the EKF update accepts.

## Two-body demo

`demo/demo_srekf.jl` loads the EKF demo's orbit, initial condition, measurements, `R`, and `Q`. The error plotted is the estimate minus the truth. Bands are ``\pm 3\sqrt{P_{ii}}`` with ``P = SS^\top``.

```julia
using SequentialFilters

filt = SquareRootExtendedKalmanFilter(0.0, x_hat0, P0)
dynamics = NonlinearODEModel(twobody!; n = n, p = μ)
process_noise_model = UnbiasedRandomProcess(1.0e-6)
measurements = NonlinearMeasurementModel(get_h_H)
for k in 1:n_meas
    predict!(filt, dynamics, process_noise_model, Δt)
    update!(filt, measurements, ys[k], R)
end
```

`dynamics` is the same augmented propagation the EKF uses, so ``\Phi`` is the state-transition matrix. The script samples that `ODESolution` for the ``\pm 3\sigma`` bands, then prints elapsed time, process CPU time, and allocations against `ExtendedKalmanFilter` on that same orbit, and against the Joseph covariance update on a 5000-step linear kernel with no ODE.

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
predict!(
    filt,
    DiscreteDynamicsModel((t, x, Δt) -> F * x; n = 2, jacobian = (t, x, dt) -> F),
    AdditiveProcessNoise(Q),
    1.0,
)
update!(filt, (t, x) -> H * x, H, [0.2], R)
(filt.x, covariance(filt))
```
