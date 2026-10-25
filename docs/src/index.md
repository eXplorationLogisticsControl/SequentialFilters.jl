# SequentialFilters.jl

Sequential filters for nonlinear state estimation.

| Filter | Stored uncertainty | Noise |
| --- | --- | --- |
| [`ExtendedKalmanFilter`](@ref) | covariance `P` | additive, through an ODE propagation |
| [`UnscentedKalmanFilter`](@ref) | covariance `P` | additive |
| [`SquareRootExtendedKalmanFilter`](@ref) | lower factor `S` with `P = S*S'` | additive |
| [`SquareRootUnscentedKalmanFilter`](@ref) | lower factor `S` with `P = S*S'` | additive |
| [`SquareRootConjugateUnscentedKalmanFilter`](@ref) | lower factor `S` with `P = S*S'` | additive, CUT weights |
| [`SquareRootCubatureKalmanFilter`](@ref) | lower factor `S` with `P = S*S'` | additive, `2n` cubature points |

The square-root filters keep a lower-triangular factor with a positive diagonal. Predict and update rebuild that factor with a thin QR, and the square-root unscented filter also uses a Cholesky rank-one downdate when a weight is negative. They do not form `P` and call `cholesky` on every step. `covariance` rebuilds `P = S*S'` when a full covariance is needed, for example to draw the `±3√P[i,i]` bands in the demos.

## Tutorials

Each tutorial states the predict and update equations, then the same calls the two-body demos make.

- [Extended Kalman filter](@ref ekf-tutorial)
- [Unscented Kalman filter](@ref ukf-tutorial)
- [Square-root EKF](@ref srekf-tutorial)
- [Square-root UKF](@ref srukf-tutorial)
- [Square-root CUT](@ref srcut-tutorial)
- [Square-root cubature Kalman filter](@ref srckf-tutorial)

## Demos and cost

The demos share the two-body problem in `demo/demo_twobody_ekf.jl`: a 7000 km orbit, range and range-rate from a rotating ground station, and the same initial covariance and process-noise scale. Each script saves an error figure (estimate minus truth, with `±3√P[i,i]` bands) and prints elapsed time, process CPU time, and allocations.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=demo demo/demo_twobody_ekf.jl
julia --project=demo demo/demo_ukf.jl
julia --project=demo demo/demo_srekf.jl
julia --project=demo demo/demo_srukf.jl
julia --project=demo demo/demo_srcut.jl
julia --project=demo demo/demo_srckf.jl
```

On the two-body timing, both members of a pair propagate the same dynamics, so that comparison includes the ODE. A second linear kernel (6 states, 2 measurements, 5000 steps, no ODE) isolates the predict and update cost. The printed `square-root / covariance` ratios are that cost. Process CPU time sums every thread, so it can exceed elapsed time when a covariance product uses more than one BLAS thread. The square-root return on these dense, low-dimensional steps is the factor: stored `S` stays positive definite. The ratios show what the QR and downdate cost relative to forming `P` directly.
