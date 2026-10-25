"""
    EnsembleSerial()

SciML ensemble algorithm that runs trajectories one after another in this
process. This is the default `ensemblealg` for every multi-point `predict!`.
"""
const EnsembleSerial = SciMLBase.EnsembleSerial

"""
    EnsembleThreads()

SciML ensemble algorithm that schedules trajectories on Julia threads.
`prob_func` hooks and in-place dynamics must be thread-safe. The process
needs more than one thread (`JULIA_NUM_THREADS`) for this to run in parallel.
"""
const EnsembleThreads = SciMLBase.EnsembleThreads

"""
    EnsembleDistributed()

SciML ensemble algorithm that schedules trajectories with `Distributed.pmap`.
Add workers with `Distributed.addprocs` before predicting. With no extra
workers the map runs in this process.
"""
const EnsembleDistributed = SciMLBase.EnsembleDistributed

"""
    EnsembleSplitThreads()

SciML ensemble algorithm that combines distributed workers with threads on
each worker. Discrete maps use the threaded path in this process. ODE
ensembles pass the algorithm through to `solve`.
"""
const EnsembleSplitThreads = SciMLBase.EnsembleSplitThreads


"""
    EnsemblePropagation(dynamics, states, t, Δt; ensemblealg=EnsembleSerial(), save_everystep=false, saveat=())

Propagate each column of `states` with `dynamics` from `t` to `t + Δt`.

`ensemblealg` selects the SciML ensemble algorithm. The default
[`EnsembleSerial`](@ref) runs one column at a time. [`EnsembleThreads`](@ref)
uses shared-memory threads. [`EnsembleDistributed`](@ref) uses `pmap`.
[`EnsembleSplitThreads`](@ref) is the split distributed/threaded algorithm.
There is no second ensemble type in this package: these are the SciML
algorithms, and this function is the propagation entry point `predict!` calls.

For a [`NonlinearODEModel`](@ref) the columns are initial conditions of one
`EnsembleProblem`, solved with the model's method and tolerances. For a
[`DiscreteDynamicsModel`](@ref) each column is one evaluation of `f(t, x, Δt)`.
`save_everystep` and `saveat` are forwarded to the ODE ensemble. A discrete
map stores the endpoints and, when `saveat` is set, the values linearly
interpolated onto those times.

Returns `(; t, X)`. `t` is the common time grid. `X[k]` is the `n × N`
state of every column at `t[k]`, with column `i` equal to input column `i`.
"""
function EnsemblePropagation(
    dynamics::NonlinearODEModel,
    states::AbstractMatrix,
    t::Float64,
    Δt::Float64;
    ensemblealg = EnsembleSerial(),
    save_everystep::Bool = false,
    saveat = (),
)
    n, N = size(states)
    if dynamics.n != n
        throw(
            DimensionMismatch(
                "ensemble has state dimension $n; dynamics model has dimension $(dynamics.n)",
            ),
        )
    end
    N >= 1 || throw(ArgumentError("ensemble propagation needs at least one state"))
    tF = t + Δt
    template = ODEProblem(dynamics.eom!, states[:, 1], (t, tF), dynamics.p)
    # `states` is read-only. Each trajectory gets its own initial condition.
    prob_func = (prob, ctx) -> remake(prob; u0 = states[:, ctx.sim_id])
    ensemble = EnsembleProblem(template; prob_func = prob_func)
    sim = if saveat == ()
        solve(
            ensemble,
            dynamics.method,
            ensemblealg;
            trajectories = N,
            reltol = dynamics.reltol,
            abstol = dynamics.abstol,
            save_everystep = save_everystep,
        )
    else
        solve(
            ensemble,
            dynamics.method,
            ensemblealg;
            trajectories = N,
            reltol = dynamics.reltol,
            abstol = dynamics.abstol,
            save_everystep = save_everystep,
            saveat = saveat,
        )
    end
    length(sim.u) == N || throw(
        ArgumentError("ensemble returned $(length(sim.u)) trajectories; expected $N"),
    )
    times = collect(Float64, sim.u[1].t)
    for sol in sim.u
        if !successful_retcode(sol)
            error("ensemble trajectory failed with retcode=$(sol.retcode)")
        end
        if abs(sol.t[end] - tF) > 1e-8 * max(1.0, abs(tF))
            error("ODE propagation ended at t=$(sol.t[end]) instead of $tF (retcode=$(sol.retcode))")
        end
    end
    return (; t = times, X = _states_on_grid(sim.u, times))
end

function EnsemblePropagation(
    dynamics::DiscreteDynamicsModel,
    states::AbstractMatrix,
    t::Float64,
    Δt::Float64;
    ensemblealg = EnsembleSerial(),
    save_everystep::Bool = false,
    saveat = (),
    )
    n, N = size(states)
    if dynamics.n != n
        throw(
            DimensionMismatch(
                "ensemble has state dimension $n; dynamics model has dimension $(dynamics.n)",
            ),
        )
    end
    N >= 1 || throw(ArgumentError("ensemble propagation needs at least one state"))
    T = eltype(states)
    X0 = Matrix{T}(states)
    Xprop = Matrix{T}(undef, n, N)
    _map_discrete!(Xprop, dynamics.f, X0, t, T(Δt), ensemblealg)
    times = _discrete_times(t, Δt, saveat, save_everystep)
    series = Vector{Matrix{T}}(undef, length(times))
    span = Δt == 0 ? one(Float64) : Float64(Δt)
    for (k, τ) in enumerate(times)
        α = (τ - t) / span
        Xk = Matrix{T}(undef, n, N)
        for i in 1:N
            for j in 1:n
                Xk[j, i] = (1 - α) * X0[j, i] + α * Xprop[j, i]
            end
        end
        series[k] = Xk
    end
    return (; t = times, X = series)
end

function EnsemblePropagation(dynamics::DynamicsModel, states::AbstractMatrix, t::Float64, Δt::Float64; kwargs...)
    throw(ArgumentError("no ensemble propagation for dynamics of type $(typeof(dynamics))"))
end


"""Propagate a point cloud and build the augmented solution. Does not add `Q`."""
function _cloud_predict(
    dynamics::DynamicsModel,
    states::AbstractMatrix,
    P_start::AbstractMatrix,
    w_mean::AbstractVector,
    w_cov::AbstractVector,
    t::Float64,
    Δt::Float64;
    ensemblealg = EnsembleSerial(),
    save_everystep::Bool = false,
    saveat = (),
)
    series = EnsemblePropagation(
        dynamics,
        states,
        t,
        Δt;
        ensemblealg = ensemblealg,
        save_everystep = save_everystep,
        saveat = saveat,
    )
    sol = _moment_solution(P_start, series, w_mean, w_cov)
    return series.X[end], sol
end


"""
`ODESolution` whose state is `[μ; vec(Φ)]`.

`Φ(t)` satisfies `Φ P_start Φᵀ = P(t)`, where `P(t)` is the weighted
covariance of the propagated points and does not include process noise.
The EKF demo reads this layout with `propagated_covariance`.
"""
function _moment_solution(P_start, series, w_mean, w_cov)
    times = series.t
    n = size(P_start, 1)
    L0 = _psd_factor(Matrix{Float64}(P_start))
    us = Vector{Vector{Float64}}(undef, length(times))
    for k in eachindex(times)
        μ = _weighted_column_mean(series.X[k], w_mean)
        P = _symmetrize(_deviation_covariance(series.X[k], μ, w_cov))
        Φ = _covariance_transition(_psd_factor(Matrix{Float64}(P)), L0)
        u = zeros(Float64, n * (n + 1))
        u[1:n] .= μ
        u[(n + 1):end] .= vec(Φ)
        us[k] = u
    end
    prob = ODEProblem((du, u, p, t) -> (du .= 0), us[1], (times[1], times[end]))
    return SciMLBase.build_solution(
        prob,
        Tsit5(),
        times,
        us;
        retcode = SciMLBase.ReturnCode.Success,
    )
end

"""
`Φ` with `Φ * L0 * L0' * Φ' = F * F'` when `L0` has full rank.

A collapsed particle set has a singular factor. The pseudoinverse keeps
`predict!` finite; the noise-free cloud covariance is then the part of
`F * F'` that lies in the column space of `L0`.
"""
function _covariance_transition(F::AbstractMatrix, L0::AbstractMatrix)
    if rank(L0) == size(L0, 1)
        return F / L0
    end
    return F * pinv(Matrix{Float64}(L0))
end

function _psd_factor(P::AbstractMatrix)
    S = _symmetrize(P)
    factorization = cholesky(Hermitian(S); check = false)
    if issuccess(factorization)
        return factorization.L
    end
    ev = eigen(Symmetric(S))
    vals = max.(ev.values, 0)
    return ev.vectors * Diagonal(sqrt.(vals))
end

function _states_on_grid(sols, times)
    n = length(sols[1].u[1])
    N = length(sols)
    T = eltype(sols[1].u[1])
    series = Vector{Matrix{T}}(undef, length(times))
    same_grid = all(sols) do sol
        length(sol.t) == length(times) && all(i -> sol.t[i] == times[i], eachindex(times))
    end
    for k in eachindex(times)
        Xk = Matrix{T}(undef, n, N)
        for i in 1:N
            state = same_grid ? sols[i].u[k] : sols[i](times[k])
            for j in 1:n
                Xk[j, i] = state[j]
            end
        end
        series[k] = Xk
    end
    return series
end

function _discrete_times(t::Float64, Δt::Float64, saveat, save_everystep::Bool)
    # A discrete map has no integrator steps. `save_everystep` does not insert times.
    save_everystep isa Bool || throw(ArgumentError("save_everystep must be a Bool"))
    tF = t + Δt
    times = Float64[t, tF]
    if saveat != ()
        for s in saveat
            τ = Float64(s)
            if t - 1e-12 <= τ <= tF + 1e-12
                push!(times, τ)
            end
        end
    end
    unique!(sort!(times))
    return times
end

function _map_discrete!(Xprop, f, states, t, Δt, ensemblealg)
    n, N = size(states)
    T = eltype(Xprop)
    function column(i)
        xi = f(t, states[:, i], Δt)
        length(xi) == n || throw(
            DimensionMismatch("f returned length $(length(xi)); expected $n"),
        )
        col = Vector{T}(undef, n)
        for j in 1:n
            col[j] = xi[j]
        end
        return col
    end
    cols = if ensemblealg isa EnsembleThreads || ensemblealg isa EnsembleSplitThreads
        out = Vector{Vector{T}}(undef, N)
        Threads.@threads for i in 1:N
            out[i] = column(i)
        end
        out
    elseif ensemblealg isa EnsembleDistributed
        Distributed.pmap(column, 1:N)
    elseif ensemblealg isa EnsembleSerial
        [column(i) for i in 1:N]
    else
        throw(
            ArgumentError(
                "discrete EnsemblePropagation supports EnsembleSerial, EnsembleThreads, EnsembleSplitThreads, and EnsembleDistributed; got $(typeof(ensemblealg))",
            ),
        )
    end
    for i in 1:N
        Xprop[:, i] = cols[i]
    end
    return Xprop
end

function _check_step(Δt::Float64)
    isfinite(Δt) || throw(ArgumentError("Δt must be finite, got $Δt"))
    return nothing
end
