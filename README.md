# `SequentialFilters.jl`: Sequential filters in Julia

<p align="center">
  <a href="https://explorationlogisticscontrol.github.io/SequentialFilters.jl">
    <img src="https://github.com/eXplorationLogisticsControl/SequentialFilters.jl/actions/workflows/docs.yml/badge.svg" alt="docs workflow"/>
  </a>
  <a href="https://github.com/eXplorationLogisticsControl/SequentialFilters.jl/actions/workflows/test.yml">
    <img src="https://github.com/eXplorationLogisticsControl/SequentialFilters.jl/actions/workflows/test.yml/badge.svg" alt="test workflow"/>
  </a>
  <!-- <a href="https://app.codecov.io/gh/eXplorationLogisticsControl/SequentialFilters.jl">
    <img src="https://codecov.io/gh/eXplorationLogisticsControl/SequentialFilters.jl/branch/main/graph/badge.svg" alt="codecov"/>
  </a> -->
</p>

<p align="center">
  <img src="demo/twobody/plots/twobody_srukf_errors.png" alt="Two-body square-root UKF estimation error" width="800"/>
</p>

Sequential filters for nonlinear state estimation. Each filter exposes the same `predict!` / `update!` loop. Covariance-form filters store `P`. Square-root filters store a lower-triangular factor `S` with `P = S*S'` and rebuild that factor with a thin QR. The [documentation](https://explorationlogisticscontrol.github.io/SequentialFilters.jl) has the predict and update equations.

## Filters

| Filter | Uncertainty | Reference |
| :--- | :--- | :--- |
| `ExtendedKalmanFilter` | covariance `P` | Kalman (1960); Joseph update of Bucy and Joseph (1968) |
| `UnscentedKalmanFilter` | covariance `P` | Wan and van der Merwe (2000) |
| `ConjugateUnscentedKalmanFilter` | covariance `P` | Adurthi, Singla, and Singh (2018) |
| `HigherOrderUnscentedEstimator` | covariance `P`, marginal skewness and kurtosis | Stojanovski and Savransky (2021) |
| `SquareRootExtendedKalmanFilter` | lower factor `S` | Kaminski, Bryson, and Schmidt (1971) |
| `SquareRootUnscentedKalmanFilter` | lower factor `S` | van der Merwe and Wan (2001) |
| `SquareRootConjugateUnscentedKalmanFilter` | lower factor `S` | Adurthi, Singla, and Singh (2018); van der Merwe and Wan (2001) |
| `SquareRootCubatureKalmanFilter` | lower factor `S` | Arasaratnam and Haykin (2009) |
| `ParticleFilter` | weighted particles | Gordon, Salmond, and Smith (1993) |

`UnscentedKalmanFilter` is the scaled unscented Kalman filter (`alpha = 1e-3`, `beta = 2`, `kappa = 0`). `ConjugateUnscentedKalmanFilter` implements CUT4 (every dimension) and CUT6 (`2 ≤ n ≤ 9`). `SquareRootCubatureKalmanFilter` is the third-degree spherical-radial rule with `2n` points. `ParticleFilter` is a bootstrap sequential importance resampling filter. `covariance` rebuilds `P = S*S'` when a square-root filter needs the full matrix.

Gaussian mixture filters and filters with consider parameters are not implemented. The consider-parameter targets are the unscented Schmidt–Kalman filter (Zanetti and DeMars, 2013; Stauch and Jah, 2015) and its square-root form (Geeraert and McMahon, 2018).

### References

- Kalman, R. E. (1960). [A New Approach to Linear Filtering and Prediction Problems](https://doi.org/10.1115/1.3662552). *Journal of Basic Engineering*, 82(1), 35–45.
- Bucy, R. S., and Joseph, P. D. (1968). *Filtering for Stochastic Processes with Applications to Guidance*. Interscience.
- Kaminski, P. G., Bryson, A. E., and Schmidt, S. F. (1971). [Discrete Square Root Filtering: A Survey of Current Techniques](https://doi.org/10.1109/TAC.1971.1099816). *IEEE Transactions on Automatic Control*, 16(6), 727–736.
- Gordon, N. J., Salmond, D. J., and Smith, A. F. M. (1993). [Novel approach to nonlinear/non-Gaussian Bayesian state estimation](https://doi.org/10.1049/ip-f-2.1993.0015). *IEE Proceedings F*, 140(2), 107–113.
- Wan, E. A., and van der Merwe, R. (2000). [The Unscented Kalman Filter for Nonlinear Estimation](https://doi.org/10.1109/ASSPCC.2000.882463). *Proc. IEEE Adaptive Systems for Signal Processing, Communications, and Control Symposium*, 153–158.
- van der Merwe, R., and Wan, E. A. (2001). [The Square-Root Unscented Kalman Filter for State and Parameter-Estimation](https://doi.org/10.1109/ICASSP.2001.940586). *Proc. IEEE ICASSP*, 3461–3464.
- Arasaratnam, I., and Haykin, S. (2009). [Cubature Kalman Filters](https://doi.org/10.1109/TAC.2009.2019800). *IEEE Transactions on Automatic Control*, 54(6), 1254–1269.
- Adurthi, N., Singla, P., and Singh, T. (2018). [Conjugate Unscented Transformation: Applications to Estimation and Control](https://doi.org/10.1115/1.4037783). *ASME Journal of Dynamic Systems, Measurement, and Control*, 140(3), 030907.
- Stojanovski, Z., and Savransky, D. (2021). [Higher-Order Unscented Estimator](https://doi.org/10.2514/1.G006109). *Journal of Guidance, Control, and Dynamics*, 44(12), 2186–2198.

## Installation

Clone this repository, change to the project that will use it, and add the local path from the package manager:

```bash
git clone https://github.com/eXplorationLogisticsControl/SequentialFilters.jl.git
cd /path/to/your/project
julia --project=.
```

```julia
pkg> add path/to/SequentialFilters.jl
```

`path/to/SequentialFilters.jl` is the directory created by `git clone`. The quick start also loads `OrdinaryDiffEq` to integrate the Lorenz equation, so add that package to the same project:

```julia
pkg> add OrdinaryDiffEq
```

## Quick start

The Lorenz-63 system is estimated from noisy measurements of `(x, z)`. The vector field, the additive process noise, and the measurement Jacobian are the same ones in [`demo/lorenz/demo_ekf.jl`](demo/lorenz/demo_ekf.jl). This snippet runs 200 steps. The demo runs 2000 steps (`Δt = 0.02`, final time 40) and writes the error figure.

```julia
using LinearAlgebra
using OrdinaryDiffEq
using Random
using SequentialFilters

function lorenz63!(du, u, p, t)
    σ, ρ, β = p
    x, y, z = u
    du[1] = σ * (y - x)
    du[2] = x * (ρ - z) - y
    du[3] = x * y - β * z
    return nothing
end

p = (10.0, 28.0, 8.0 / 3.0)   # σ, ρ, β
Δt = 0.02
n_steps = 200
x0 = [1.0, 1.0, 1.0]          # off the attractor

truth = solve(
    ODEProblem(lorenz63!, x0, (0.0, n_steps * Δt), p),
    Tsit5();
    reltol = 1e-10,
    abstol = 1e-10,
    saveat = Δt,
)

σ_meas = 1.0
R = Diagonal(fill(σ_meas^2, 2))
Random.seed!(1)
ys = map(1:n_steps) do k
    state = truth(k * Δt)
    return [state[1], state[3]] .+ σ_meas .* randn(2)
end

dynamics = NonlinearODEModel(lorenz63!; n = 3, p = p, reltol = 1e-5, abstol = 1e-5)
process_noise = AdditiveProcessNoise(Matrix(1.0e-3 * I(3)))

h(t, x) = [x[1], x[3]]
function get_h_H(t, x)
    H = zeros(2, 3)
    H[1, 1] = 1.0
    H[2, 3] = 1.0
    return h(t, x), H
end
measurements = NonlinearMeasurementModel(get_h_H)

x_hat0 = x0 .+ [5.0, -5.0, 8.0]
P0 = Matrix(Diagonal(fill(25.0, 3)))
filt = ExtendedKalmanFilter(0.0, x_hat0, P0)

for k in 1:n_steps
    predict!(filt, dynamics, process_noise, Δt)
    update!(filt, measurements, ys[k], R)
end

println("estimate: ", filt.x)
println("truth:    ", truth(filt.t))
println("error:    ", filt.x .- truth(filt.t))
```

`predict!` integrates the continuous dynamics and the state-transition matrix, then adds `Q`. `update!` applies the measurement Jacobian and the Joseph covariance update. Both mutate `filt`.

Derivative-free filters take the measurement function instead of the Jacobian model. The same loop with an unscented filter is `filt = UnscentedKalmanFilter(0.0, x_hat0, P0)` and `update!(filt, h, ys[k], R)`. The Lorenz and two-body scripts under `demo/` cover every filter in the table, and the [two-body tutorial](https://explorationlogisticscontrol.github.io/SequentialFilters.jl/dev/tutorials/basic/) walks through a range and range-rate example.

```bash
julia --project=demo demo/lorenz/demo_ekf.jl
julia --project=demo demo/twobody/demo_ekf.jl
```
