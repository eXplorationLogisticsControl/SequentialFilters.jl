# [Bootstrap particle filter (PF)](@id pf-tutorial)

`ParticleFilter` is a bootstrap sequential importance resampling filter with additive process and measurement noise. Particles are stored as columns of `particles`. `weights` sum to 1. `x` and `P` are the weighted sample mean and covariance (maximum-likelihood weights, not `1/(N-1)`).

The constructor draws `n_particles` samples from `N(x, P)`. The default is 100. `resampling` is `:systematic` (the default) or `:multinomial`. Both draw exactly `n_particles` ancestors, so a resampling step does not change the count. `rng` supplies the initial draw, the process-noise samples, and the resampling. The two-body demo is `demo/twobody/demo_particle.jl` with 64 particles.

## Motivation

A Gaussian filter, including the [EKF](@ref ekf-tutorial) and the [UKF](@ref ukf-tutorial), replaces the filtering distribution by a mean and a covariance. That is a poor description when the posterior ``p(x_{k|k})`` is skewed or has more than one mode. A particle filter keeps a weighted set of states. The bootstrap filter pushes those states through the dynamics and reweights them by the measurement likelihood.

The price is the number of trajectories. The default propagation is serial. Pass an ensemble algorithm when the dynamics are expensive enough that the points should run together.

**Further reading.**

- Gordon, N. J., Salmond, D. J., and Smith, A. F. M. (1993). [Novel approach to nonlinear/non-Gaussian Bayesian state estimation](https://doi.org/10.1049/ip-f-2.1993.0015). *IEE Proceedings F*, 140(2), 107–113.
- Arulampalam, M. S., Maskell, S., Gordon, N., and Clapp, T. (2002). [A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking](https://doi.org/10.1109/78.978374). *IEEE Transactions on Signal Processing*, 50(2), 174–188.

## Prediction

Each particle is an independent draw. With additive process noise,

```math
x_{k|k-1}^{(i)} = f\bigl(t_{k-1}, x_{k-1|k-1}^{(i)}, \Delta t\bigr) + w_k^{(i)},
\qquad w_k^{(i)} \sim \mathcal N(0, Q).
```

Weights are unchanged. The reported mean and covariance are the weighted sample moments of the new set, and time advances by ``\Delta t``.

`predict!` takes a [`DynamicsModel`](@ref) and a [`ProcessNoiseModel`](@ref). ``Q`` is `get_noise(process_noise_model, Δt)`. The noise-free map is [`EnsemblePropagation`](@ref). The default `ensemblealg` is [`EnsembleSerial`](@ref). Pass [`EnsembleThreads`](@ref), [`EnsembleDistributed`](@ref), or [`EnsembleSplitThreads`](@ref) to change the parallelism. Process-noise samples are drawn after that propagation, on `filter.rng`, so the random stream is not shared across threads. The returned `ODESolution` stores `[μ; vec(Φ)]` along the noise-free cloud, with ``\Phi P_{k-1|k-1} \Phi^\top`` equal to that cloud's covariance. `filter.x` and `filter.P` include the process-noise samples.

## Update

The measurement density is Gaussian. For a [`MeasurementModel`](@ref), `get_h_H` supplies ``\hat y_{k|k-1}`` and the Jacobian is ignored. A noise-free `h(t, x)` is also accepted.

```math
w_{k|k}^{(i)} \propto w_{k|k-1}^{(i)}\,
\mathcal N\bigl(y_k;\, h(t_k, x_{k|k-1}^{(i)}),\, R\bigr).
```

Weights are normalized so they sum to 1. [`resample!`](@ref) then replaces the particles. `:systematic` uses one uniform draw and equally spaced ranks. `:multinomial` draws each ancestor independently. The new weights are `1/N`.

## Two-body demo

`demo/twobody/demo_particle.jl` uses the EKF demo's orbit, initial condition, range and range-rate measurements, and noise. The filter has 64 particles and a fixed random stream. The figure samples the returned `ODESolution` and draws the estimation error with bands ``\pm 3\sqrt{P_{ii}}``. The figure is `demo/twobody/plots/twobody_particle_errors.png`.

```julia
using SequentialFilters

filt = ParticleFilter(0.0, x_hat0, P0; n_particles = 64, rng = MersenneTwister(1))
dynamics = NonlinearODEModel(twobody!; n = n, p = μ)
process_noise_model = UnbiasedRandomProcess(1.0e-6)
measurements = NonlinearMeasurementModel(get_h_H)
for k in 1:n_meas
    predict!(filt, dynamics, process_noise_model, Δt)
    update!(filt, measurements, ys[k], R)
end
```

The default `ensemblealg` is `EnsembleSerial()`. Pass `EnsembleThreads()`, `EnsembleDistributed()`, or `EnsembleSplitThreads()` on `predict!` to propagate the particles in parallel.

The script then prints elapsed time, process CPU time, and allocations for a 64-particle two-body run and a 32-particle linear kernel of 200 steps.

```julia
print_filter_costs(case, :particle)
```

```bash
julia --project=demo demo/twobody/demo_particle.jl
```

## Same calls on a linear model

```@example pf-step
using LinearAlgebra
using Random
using SequentialFilters

F = [1.0 0.5; 0.0 1.0]
H = [1.0 0.0]
Q = [1.0e-3 0.0; 0.0 1.0e-3]
R = [0.05;;]
filt = ParticleFilter(0.0, [0.0, 1.0], Matrix{Float64}(I, 2, 2); n_particles = 32, rng = MersenneTwister(1))
predict!(filt, DiscreteDynamicsModel((t, x, Δt) -> F * x; n = 2), AdditiveProcessNoise(Q), 1.0)
update!(filt, (t, x) -> H * x, [0.2], R)
(filt.x, sum(filt.weights), size(filt.particles, 2))
```
