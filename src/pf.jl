"""
    ParticleFilter(t, x, P; n_particles=100, resampling=:systematic, rng=Random.default_rng())

Bootstrap sequential-importance-resampling filter with additive process and
measurement noise.

`n_particles` independent draws from `N(x, P)` are stored as columns of
`particles`. `weights` are uniform and sum to 1. `x` and `P` are then the
weighted sample mean and covariance of those particles, so they can move off
the constructor arguments by a sampling fluctuation.

`resampling` is `:systematic` (the default) or `:multinomial`. Both draw
exactly `n_particles` ancestors, so a resampling step does not change the
particle count. `rng` is used for the initial draw, the process noise, and
the resampling.

`predict!` propagates every particle with [`EnsemblePropagation`](@ref) and
then adds one sample of [`get_noise`](@ref). Weights are unchanged.
`update!` reweights by a Gaussian measurement likelihood and resamples.
The measurement can be a [`MeasurementModel`](@ref), the same object the
extended Kalman filter update accepts: only the predicted measurement is
used, not the Jacobian. A noise-free `h(t, x)` is also accepted.
"""
mutable struct ParticleFilter
    t::Float64
    particles::Matrix{Float64}
    weights::Vector{Float64}
    x::Vector{Float64}
    P::Matrix{Float64}
    nx::Int
    resampling::Symbol
    rng::Random.AbstractRNG

    function ParticleFilter(
        t::Real,
        x::AbstractVector{<:Real},
        P::AbstractMatrix{<:Real};
        n_particles::Integer = 100,
        resampling::Symbol = :systematic,
        rng::Random.AbstractRNG = Random.default_rng(),
    )
        nx = length(x)
        nx >= 1 || throw(ArgumentError("state dimension must be positive, got $nx"))
        if size(P) != (nx, nx)
            throw(DimensionMismatch("P is $(size(P)); expected ($nx, $nx)"))
        end
        if resampling != :systematic && resampling != :multinomial
            throw(ArgumentError("resampling must be :systematic or :multinomial, got $resampling"))
        end
        N = Int(n_particles)
        N >= 1 || throw(ArgumentError("n_particles must be positive, got $N"))
        L = _lower_cholesky(Matrix{Float64}(P))
        x0 = collect(Float64, x)
        particles = Matrix{Float64}(undef, nx, N)
        for i in 1:N
            particles[:, i] = x0 .+ L * randn(rng, nx)
        end
        weights = fill(1 / N, N)
        filter = new(Float64(t), particles, weights, copy(x0), Matrix{Float64}(I, nx, nx), nx, resampling, rng)
        _refresh_particle_moments!(filter)
        return filter
    end
end


"""
    covariance(filter::ParticleFilter)

Weighted sample covariance of the particles. This is the same matrix stored
in `filter.P`.
"""
function covariance(filter::ParticleFilter)
    return filter.P
end


"""
    predict!(filter::ParticleFilter, dynamics_model, process_noise_model, Δt; ensemblealg=EnsembleSerial(), save_everystep=false, saveat=())

Propagate every particle with [`EnsemblePropagation`](@ref) and add one draw
of `N(0, Q)`, where `Q = get_noise(process_noise_model, Δt)`.

The default `ensemblealg` is [`EnsembleSerial`](@ref). Pass
[`EnsembleThreads`](@ref), [`EnsembleDistributed`](@ref), or
[`EnsembleSplitThreads`](@ref) to change the parallelism. Weights are held
fixed. The returned `ODESolution` stores `[μ; vec(Φ)]` along the noise-free
cloud, with `Φ P_start Φᵀ` equal to that cloud's covariance. Process noise is
applied to the particles after that solution is built, and `filter.x` /
`filter.P` include it.
"""
function predict!(
    filter::ParticleFilter,
    dynamics_model::DynamicsModel,
    process_noise_model::ProcessNoiseModel,
    Δt::Float64;
    save_everystep::Bool = false,
    saveat = (),
    ensemblealg = EnsembleSerial(),
)
    _check_step(Δt)
    Q = _as_eltype(_square_matrix(get_noise(process_noise_model, Δt), filter.nx, "Q"), Float64)
    P_start = copy(filter.P)
    Xfinal, sol = _cloud_predict(
        dynamics_model,
        filter.particles,
        P_start,
        filter.weights,
        filter.weights,
        filter.t,
        Δt;
        ensemblealg = ensemblealg,
        save_everystep = save_everystep,
        saveat = saveat,
    )
    factor = _noise_factor(Q)
    nnoise = size(factor, 2)
    for i in axes(Xfinal, 2)
        particle = Xfinal[:, i]
        if nnoise > 0
            particle = particle .+ factor * randn(filter.rng, nnoise)
        end
        filter.particles[:, i] = particle
    end
    filter.t = filter.t + Δt
    _refresh_particle_moments!(filter)
    return sol
end


"""
    update!(filter::ParticleFilter, measurement_model::MeasurementModel, y, R)
    update!(filter::ParticleFilter, h, y, R)

Gaussian measurement update, then resampling.

For a [`MeasurementModel`](@ref), `get_h_H` supplies the predicted
measurement and the Jacobian is ignored. `h(t, x)` is the noise-free
measurement used by the unscented filters. The likelihood is
`N(y; yhat, R)`. Weights are normalized so they sum to 1, then
[`resample!`](@ref) replaces the particles and resets the weights to `1/N`.
"""
function update!(filter::ParticleFilter, model::MeasurementModel, y, R)
    return _particle_update!(filter, x -> model.get_h_H(filter.t, x)[1], y, R)
end

function update!(filter::ParticleFilter, h, y, R)
    return _particle_update!(filter, x -> h(filter.t, x), y, R)
end


"""
    resample!(filter::ParticleFilter)

Draw a new set of particles with replacement. The new set has the same count.
`:systematic` uses one uniform draw and equally spaced ranks. `:multinomial`
draws each ancestor independently. Weights are reset to `1/N`.
"""
function resample!(filter::ParticleFilter)
    N = size(filter.particles, 2)
    weights = filter.weights
    idx = if filter.resampling == :multinomial
        _multinomial_indices(weights, filter.rng)
    else
        _systematic_indices(weights, filter.rng)
    end
    length(idx) == N || error("resampling produced $(length(idx)) indices; expected $N")
    old = copy(filter.particles)
    for i in 1:N
        filter.particles[:, i] = old[:, idx[i]]
    end
    filter.weights .= 1 / N
    _refresh_particle_moments!(filter)
    return filter
end


function _particle_update!(filter::ParticleFilter, predicted, y, R)
    yv = _measurement_vector(y)
    ny = length(yv)
    Rm = _square_matrix(R, ny, "R")
    factor = cholesky(Hermitian(_symmetrize(Rm)))
    N = size(filter.particles, 2)
    logw = Vector{Float64}(undef, N)
    for i in 1:N
        yhat = collect(Float64, predicted(filter.particles[:, i]))
        length(yhat) == ny || throw(
            DimensionMismatch("measurement model returned length $(length(yhat)); expected $ny"),
        )
        innov = yv .- yhat
        z = factor.L \ innov
        logw[i] = log(filter.weights[i]) - 0.5 * dot(z, z)
    end
    m = maximum(logw)
    isfinite(m) || throw(ArgumentError("measurement update produced non-finite weights"))
    scaled = exp.(logw .- m)
    total = sum(scaled)
    total > 0 || throw(ArgumentError("measurement update produced zero weights"))
    filter.weights .= scaled ./ total
    resample!(filter)
    return nothing
end

function _refresh_particle_moments!(filter::ParticleFilter)
    μ, P = _particle_moments(filter.particles, filter.weights)
    filter.x .= μ
    filter.P .= P
    return nothing
end

function _particle_moments(particles::AbstractMatrix, weights::AbstractVector)
    n, N = size(particles)
    μ = zeros(Float64, n)
    for i in 1:N
        wi = weights[i]
        for j in 1:n
            μ[j] += wi * particles[j, i]
        end
    end
    P = zeros(Float64, n, n)
    for i in 1:N
        d = particles[:, i] .- μ
        P .+= weights[i] .* (d * d')
    end
    return μ, _symmetrize(P)
end

function _systematic_indices(weights::AbstractVector, rng)
    N = length(weights)
    cumulative = cumsum(weights)
    cumulative[end] = 1
    u0 = rand(rng) / N
    idx = Vector{Int}(undef, N)
    i = 1
    for j in 1:N
        u = u0 + (j - 1) / N
        while i < N && cumulative[i] < u
            i += 1
        end
        idx[j] = i
    end
    return idx
end

function _multinomial_indices(weights::AbstractVector, rng)
    N = length(weights)
    cumulative = cumsum(weights)
    cumulative[end] = 1
    idx = Vector{Int}(undef, N)
    for j in 1:N
        i = searchsortedfirst(cumulative, rand(rng))
        idx[j] = min(max(i, 1), N)
    end
    return idx
end
