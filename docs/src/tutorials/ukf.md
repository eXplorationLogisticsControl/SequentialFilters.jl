# [Unscented Kalman filter](@id ukf-tutorial)

`UnscentedKalmanFilter` is the scaled unscented Kalman filter of Wan and van der Merwe, with additive process and measurement noise. It stores the mean `x` and the covariance `P`. No Jacobians are required.

Defaults are `alpha = 1e-3`, `beta = 2`, and `kappa = 0`. Weights come from [`unscented_weights`](@ref). The two-body demo is `demo/demo_ukf.jl`.

## Motivation

The [EKF](@ref ekf-tutorial) replaces a nonlinear map by its Jacobian at the mean. That first-order model is wrong as soon as the covariance is large enough for the higher derivatives to matter, and some maps are awkward to differentiate. The unscented transform instead draws a small deterministic set of sigma points from the current ``(x, P)``, pushes each point through the true ``f`` or ``h``, and rebuilds the mean and covariance from those images. For a Gaussian prior the transformed mean is correct through third order and the covariance through at least second order, with no Jacobian.

Use the UKF when derivatives are unavailable, expensive, or a poor description of the map across the current ``P``. The price is ``2n+1`` evaluations of ``f`` or ``h`` per step. The filter is still a Gaussian approximation: it does not represent multimodality. Small `alpha` (the default `1e-3`) keeps the points near the mean and behaves like a derivative expansion; a larger `alpha` (up to about 1) is appropriate when the map is strongly nonlinear across ``P``. `beta = 2` is the Gaussian kurtosis choice.

**Further reading.**

- Julier, S. J., and Uhlmann, J. K. (1997). A New Extension of the Kalman Filter to Nonlinear Systems. *Proc. SPIE* 3068, 182–193.
- Julier, S. J., Uhlmann, J. K., and Durrant-Whyte, H. F. (2000). [A New Method for the Nonlinear Transformation of Means and Covariances in Filters and Estimators](https://doi.org/10.1109/9.847726). *IEEE Transactions on Automatic Control*, 45(3), 477–482.
- Wan, E. A., and van der Merwe, R. (2000). [The Unscented Kalman Filter for Nonlinear Estimation](https://doi.org/10.1109/ASSPCC.2000.882463). *Proc. IEEE Adaptive Systems for Signal Processing, Communications, and Control Symposium*, 153–158.
- Julier, S. J., and Uhlmann, J. K. (2004). [Unscented Filtering and Nonlinear Estimation](https://doi.org/10.1109/JPROC.2003.823141). *Proceedings of the IEEE*, 92(3), 401–422.

## Prediction

For state dimension ``n``,

```math
\lambda = \alpha^2 (n + \kappa) - n, \qquad c = n + \lambda = \alpha^2 (n + \kappa).
```

Mean and covariance weights, with Julia index `1` equal to the center ``i = 0``,

```math
W_m^{(0)} = \lambda / c, \qquad
W_c^{(0)} = \lambda / c + (1 - \alpha^2 + \beta),
```

```math
W_m^{(i)} = W_c^{(i)} = \frac{1}{2c}, \qquad i = 1,\ldots,2n.
```

Sigma points use the lower Cholesky factor ``P_{k-1:k-1} = LL^\top``:

```math
\chi_{k-1:k-1}^{(0)} = x_{k-1:k-1}, \qquad
\chi_{k-1:k-1}^{(i)} = x_{k-1:k-1} + \sqrt{c}\, L_{:,i},
```

```math
\chi_{k-1:k-1}^{(n+i)} = x_{k-1:k-1} - \sqrt{c}\, L_{:,i}.
```

Each point is passed through the noise-free model ``f(t, x, \Delta t)``:

```math
x_{k:k-1} = \sum_{i=0}^{2n} W_m^{(i)}\, f\bigl(t_{k-1}, \chi_{k-1:k-1}^{(i)}, \Delta t\bigr),
```

```math
P_{k:k-1} = \sum_{i=0}^{2n} W_c^{(i)}
\big(f\bigl(t_{k-1}, \chi_{k-1:k-1}^{(i)}, \Delta t\bigr) - x_{k:k-1}\big)
\big(f\bigl(t_{k-1}, \chi_{k-1:k-1}^{(i)}, \Delta t\bigr) - x_{k:k-1}\big)^\top + Q.
```

``P_{k:k-1}`` is symmetrized, and time advances by ``\Delta t``.

## Update

Sigma points are redrawn from the prior ``\bigl(x_{k:k-1}, P_{k:k-1}\bigr)`` and passed through ``h(t, x)``:

```math
\hat y_{k:k-1} = \sum_{i=0}^{2n} W_m^{(i)}\, h\bigl(t_k, \chi_{k:k-1}^{(i)}\bigr),
```

```math
S_{y,k:k-1} = \sum_{i=0}^{2n} W_c^{(i)}
\big(h\bigl(t_k, \chi_{k:k-1}^{(i)}\bigr) - \hat y_{k:k-1}\big)
\big(h\bigl(t_k, \chi_{k:k-1}^{(i)}\bigr) - \hat y_{k:k-1}\big)^\top + R,
```

```math
P_{xy,k:k-1} = \sum_{i=0}^{2n} W_c^{(i)}
\big(\chi_{k:k-1}^{(i)} - x_{k:k-1}\big)
\big(h\bigl(t_k, \chi_{k:k-1}^{(i)}\bigr) - \hat y_{k:k-1}\big)^\top,
```

```math
K = P_{xy,k:k-1} S_{y,k:k-1}^{-1}, \qquad
x_{k:k} = x_{k:k-1} + K\bigl(y_{k} - \hat y_{k:k-1}\bigr),
```

```math
P_{k:k} = P_{k:k-1} - K S_{y,k:k-1} K^\top.
```

``P_{k:k}`` is symmetrized. This is the additive-noise form, not a joint augmented state.

## Two-body demo

`demo/demo_ukf.jl` uses the dynamics, initial mean and covariance, range and range-rate measurements, and noise from the EKF demo. `f` integrates the two-body ODE with the same state-transition propagation as `propagate!`, and `h` is the ground-station measurement. The figure is the estimation error (estimate minus truth) with bands ``\pm 3\sqrt{P_{ii}}``.

```julia
using SequentialFilters

include("twobody_scenario.jl")
case = load_twobody_case()
ukf = UnscentedKalmanFilter(0.0, case.x_hat0, case.P0)
for k in 1:case.n_meas
    predict!(ukf, case.f, case.Q, case.Δt)
    update!(ukf, case.h, case.ys[k], case.R)
end
```

The same script then times this filter against the square-root UKF. Both two-body runs propagate every sigma point. The linear kernel repeats a 6-state, 2-measurement model for 5000 steps with ``f(x) = Fx`` and ``h(x) = Hx``, which is the predict/update cost without the ODE.

```julia
print_filter_costs(case, :ukf)
```

```bash
julia --project=demo demo/demo_ukf.jl
```

## Same calls on a linear model

```@example ukf-step
using LinearAlgebra
using SequentialFilters

F = [1.0 0.5; 0.0 1.0]
H = [1.0 0.0]
Q = [1.0e-3 0.0; 0.0 1.0e-3]
R = [0.05;;]
ukf = UnscentedKalmanFilter(0.0, [0.0, 1.0], Matrix{Float64}(I, 2, 2))
predict!(ukf, (t, x, Δt) -> F * x, Q, 1.0)
update!(ukf, (t, x) -> H * x, [0.2], R)
ukf.x
```
