"""
Scenario interface for a Monte Carlo filter comparison.

A scenario draws one truth, one initial estimate, and one measurement sequence.
Filters built from that draw share it. `f` is the filter's noise-free discrete
map. `f_variational` and `F` are the paired state and state-transition matrix
used by the square-root EKF. `f_truth` and `F_truth` are the matched model
used for the truth and the posterior Cramér–Rao bound.
"""

struct Scenario
    name::String
    nx::Int
    ny::Int
    n_steps::Int
    Δt::Float64
    blocks::NamedTuple
    divergence_threshold::Float64
    divergence_indices::UnitRange{Int}
    P0::Matrix{Float64}
    Q::Matrix{Float64}
    R::Matrix{Float64}
    f::Any
    f_variational::Any
    F::Any
    h::Any
    H::Any
    dynamics::Any
    process_noise::Any
    measurements::Any
    draw::Any
    reference_truth::Matrix{Float64}
    f_truth::Any
    F_truth::Any
    H_truth::Any
    times::Vector{Float64}
    parameters::NamedTuple
end

"""Noise-free trajectory `x_{k+1} = f(t_k, x_k, Δt)`, including the initial state."""
function noise_free_trajectory(f, x0, n_steps::Integer, Δt::Real)
    nx = length(x0)
    X = Matrix{Float64}(undef, nx, Int(n_steps) + 1)
    x = collect(Float64, x0)
    X[:, 1] = x
    t = 0.0
    dt = Float64(Δt)
    for k in 1:Int(n_steps)
        x = collect(Float64, f(t, x, dt))
        X[:, k + 1] = x
        t += dt
    end
    return X
end

"""Callable one-step state map. The type is shared by every parameter value."""
struct StateStep{E,P,Pr}
    eom!::E
    p::P
    base::Pr
    reltol::Float64
    abstol::Float64
end

function StateStep(eom!, p, n::Integer; reltol::Float64 = 1e-12, abstol::Float64 = 1e-12)
    base = ODEProblem(eom!, zeros(Float64, Int(n)), (0.0, 1.0), p)
    return StateStep(eom!, p, base, reltol, abstol)
end

function (step::StateStep)(t, x, Δt)
    x64 = collect(Float64, x)
    t64 = Float64(t)
    dt64 = Float64(Δt)
    tF = t64 + dt64
    sol = solve(
        remake(step.base; u0 = x64, tspan = (t64, tF), p = step.p),
        Tsit5();
        reltol = step.reltol,
        abstol = step.abstol,
        save_everystep = false,
    )
    if abs(sol.t[end] - tF) > 1e-8 * max(1.0, abs(tF))
        error("ODE propagation ended at t=$(sol.t[end]) instead of $tF (retcode=$(sol.retcode))")
    end
    return copy(sol.u[end])
end

struct VariationalEvaluation{D,N}
    dynamics::D
    process_noise::N
    cache::Ref{Any}
end

struct VariationalState{D,N}
    evaluation::VariationalEvaluation{D,N}
end

struct VariationalJacobian{D,N}
    evaluation::VariationalEvaluation{D,N}
end

function _variational_step(evaluation::VariationalEvaluation, t, x, Δt)
    n = evaluation.dynamics.n
    x64 = collect(Float64, x)
    t64 = Float64(t)
    dt64 = Float64(Δt)
    hit = evaluation.cache[]
    if hit !== nothing && hit.t == t64 && hit.Δt == dt64 && hit.x == x64
        evaluation.cache[] = nothing
        return copy(hit.xnext), copy(hit.Φ)
    end
    xwork = copy(x64)
    Pwork = Matrix{Float64}(I, n, n)
    _, sol = SequentialFilters.propagate!(
        evaluation.dynamics,
        evaluation.process_noise,
        t64,
        dt64,
        xwork,
        Pwork,
    )
    Φ = reshape(copy(sol.u[end][(n + 1):end]), n, n)
    xnext = copy(xwork)
    evaluation.cache[] = (; t = t64, Δt = dt64, x = x64, xnext = xnext, Φ = Φ)
    return copy(xnext), copy(Φ)
end

(step::VariationalState)(t, x, Δt) = _variational_step(step.evaluation, t, x, Δt)[1]
(step::VariationalJacobian)(t, x, Δt) = _variational_step(step.evaluation, t, x, Δt)[2]

"""
Paired noise-free state map and state-transition matrix.

The square-root EKF calls the state map and then the Jacobian on the same
state. The two callables share one variational integration. Process noise is
applied by the filter, not by this map.
"""
function variational_pair(dynamics, process_noise)
    evaluation = VariationalEvaluation(dynamics, process_noise, Ref{Any}(nothing))
    return VariationalState(evaluation), VariationalJacobian(evaluation)
end
