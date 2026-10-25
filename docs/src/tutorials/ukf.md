# [Unscented Kalman filter](@id ukf-tutorial)

`UnscentedKalmanFilter` is the scaled unscented Kalman filter of Wan and van der Merwe, with additive process and measurement noise. It stores the mean `x` and the covariance `P`. No Jacobians are required.

Defaults are `alpha = 1e-3`, `beta = 2`, and `kappa = 0`. Weights come from [`unscented_weights`](@ref). The two-body demo is `demo/demo_ukf.jl`.

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

Sigma points use the lower Cholesky factor ``P = LL^\top``:

```math
\chi^{(0)} = x, \qquad
\chi^{(i)} = x + \sqrt{c}\, L_{:,i}, \qquad
\chi^{(n+i)} = x - \sqrt{c}\, L_{:,i}.
```

Each point is passed through the noise-free model ``f(t, x, \Delta t)``:

```math
\hat x = \sum_{i=0}^{2n} W_m^{(i)}\, f(t, \chi^{(i)}, \Delta t),
```

```math
P = \sum_{i=0}^{2n} W_c^{(i)}
\big(f(t, \chi^{(i)}, \Delta t) - \hat x\big)
\big(f(t, \chi^{(i)}, \Delta t) - \hat x\big)^\top + Q.
```

`P` is symmetrized, and `t` advances by ``\Delta t``.

## Update

Sigma points are redrawn from the predicted mean and covariance and passed through ``h(t, x)``:

```math
\hat y = \sum_{i=0}^{2n} W_m^{(i)}\, h(t, \chi^{(i)}),
```

```math
S = \sum_{i=0}^{2n} W_c^{(i)}
\big(h(t, \chi^{(i)}) - \hat y\big)
\big(h(t, \chi^{(i)}) - \hat y\big)^\top + R,
```

```math
P_{xy} = \sum_{i=0}^{2n} W_c^{(i)}
\big(\chi^{(i)} - \hat x\big)
\big(h(t, \chi^{(i)}) - \hat y\big)^\top,
```

```math
K = P_{xy} S^{-1}, \qquad
x \leftarrow x + K(y - \hat y), \qquad
P \leftarrow P - K S K^\top.
```

`P` is symmetrized. This is the additive-noise form, not a joint augmented state.

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
