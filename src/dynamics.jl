"""
Process-noise model.

`get_noise(model, Δt)` is the discrete covariance added over a step of length `Δt`.
"""
abstract type ProcessNoiseModel end


"""
    UnbiasedRandomProcess(σp)

White-acceleration process noise for a 6-state position/velocity model.

`σp` is the acceleration scale. `get_noise` returns the matching discrete covariance.
"""
mutable struct UnbiasedRandomProcess <: ProcessNoiseModel
    σp::Float64

    function UnbiasedRandomProcess(σp::Float64)
        return new(σp)
    end
end


"""
    get_noise(model::UnbiasedRandomProcess, Δt::Float64)

Discrete process-noise covariance over an interval of length `Δt`:

```math
Q = \\sigma_p^2
\\begin{bmatrix}
(\\Delta t^3/3) I_3 & (\\Delta t^2/2) I_3 \\\\
(\\Delta t^2/2) I_3 & \\Delta t\\, I_3
\\end{bmatrix}.
```
"""
function get_noise(noise_model::UnbiasedRandomProcess, Δt::Float64)
    return noise_model.σp^2 * [
        Δt^3/3 * I(3) Δt^2/2 * I(3)
        Δt^2/2 * I(3) Δt * I(3)
    ]
end


"""
    AdditiveProcessNoise(Q)

Discrete process-noise covariance added at the end of one predict step.

`Q` is an `n × n` covariance (`Matrix` or `Diagonal`). [`get_noise`](@ref)
returns that matrix and does not scale it by `Δt`. Use this for a discrete
map. [`UnbiasedRandomProcess`](@ref) remains the 6-state continuous model
whose discrete covariance grows with `Δt`.
"""
struct AdditiveProcessNoise{M<:AbstractMatrix{<:Real}} <: ProcessNoiseModel
    Q::M

    function AdditiveProcessNoise(Q::AbstractMatrix{<:Real})
        if size(Q, 1) != size(Q, 2)
            throw(DimensionMismatch("Q is $(size(Q)); expected a square covariance"))
        end
        return new{typeof(Q)}(Q)
    end
end

function get_noise(noise_model::AdditiveProcessNoise, Δt::Float64)
    isfinite(Δt) || throw(ArgumentError("Δt must be finite, got $Δt"))
    return noise_model.Q
end


"""Continuous-time dynamics integrated over one filter predict step."""
abstract type DynamicsModel end

"""
Nonlinear dynamics `eom!(du, u, p, t)` integrated with OrdinaryDiffEq.

`n` is the state dimension. `p` is the ODE parameter. `method`, `reltol`, and
`abstol` are passed to `solve`. `augmented!` integrates the state together with
its state-transition matrix, and `base_ode_augmented` is `remake`d on each
propagation. `eom!` and `n` are fixed at construction; `p`, `method`, `reltol`,
and `abstol` are read on each call.
"""
mutable struct NonlinearODEModel{E,P,M,A,Pr} <: DynamicsModel
    eom!::E
    p::P
    method::M
    reltol::Float64
    abstol::Float64
    n::Int
    augmented!::A
    base_ode_augmented::Pr
end

function NonlinearODEModel(
    eom!;
    n::Integer,
    p = nothing,
    method = Tsit5(),
    reltol::Float64 = 1e-12,
    abstol::Float64 = 1e-12,
)
    n = Int(n)
    augmented! = let n = n, eom! = eom!
        function augmented!(du, u, p_ode, t_ode)
            xk = @view u[1:n]
            Φ = reshape(@view(u[(n + 1):end]), n, n)
            eom!(@view(du[1:n]), xk, p_ode, t_ode)
            F = _dynamics_jacobian(eom!, xk, p_ode, t_ode)
            du[(n + 1):end] .= vec(F * Φ)
            return nothing
        end
    end
    u0 = zeros(n * (n + 1))
    prob = ODEProblem(augmented!, u0, (0.0, 1.0), p)
    return NonlinearODEModel(eom!, p, method, reltol, abstol, n, augmented!, prob)
end


"""
    DiscreteDynamicsModel(f; n, jacobian=nothing)

Noise-free discrete map `f(t, x, Δt)` of an `n`-dimensional state.

Sigma-point and particle filters push each point through `f`. The
square-root EKF also needs the Jacobian of `f`. Pass `jacobian(t, x, Δt)`,
or leave it unset and the square-root EKF builds that matrix with ForwardDiff.
`jacobian` is not used by the unscented, CUT, cubature, or particle filters.
"""
struct DiscreteDynamicsModel{F,J} <: DynamicsModel
    f::F
    jacobian::J
    n::Int

    function DiscreteDynamicsModel(f; n::Integer, jacobian = nothing)
        n_int = Int(n)
        n_int >= 1 || throw(ArgumentError("state dimension must be positive, got $n_int"))
        return new{typeof(f), typeof(jacobian)}(f, jacobian, n_int)
    end
end

"""
Jacobian of a discrete map. A stored `jacobian` is called as
`jacobian(t, x, Δt)`. Otherwise the columns are ForwardDiff derivatives of `f`.
"""
function discrete_jacobian(model::DiscreteDynamicsModel, t, x, Δt)
    if model.jacobian !== nothing
        return model.jacobian(t, x, Δt)
    end
    x_eval = x isa Vector ? x : collect(x)
    return ForwardDiff.jacobian(x_eval) do z
        return collect(model.f(t, z, Δt))
    end
end

"""
Jacobian of an in-place OrdinaryDiffEq right-hand side at state `x`.
"""
function _dynamics_jacobian(eom!, x, p, t)
    x_eval = x isa Vector ? x : collect(x)
    return ForwardDiff.jacobian(x_eval) do z
        dz = similar(z)
        eom!(dz, z, p, t)
        return dz
    end
end

"""
Solve the state and state-transition matrix over `[t, t + Δt]`.

Writes the propagated state into `x` and replaces `P` with the symmetrized
`Φ P Φᵀ` plus process noise. Returns `(t + Δt, sol)`.

`sol.u` stores the augmented state `[x; vec(Φ)]`, where `Φ` is the
state-transition matrix from the start of this interval (column-major).
`saveat` and `save_everystep` are passed to `solve`; the default stores only
the interval endpoints. Covariance of the continuous propagation along the
solution is `Φ(t) P_start Φ(t)ᵀ`. Process noise is added once, when `P` is
updated at the end of the step.
"""
function propagate!(
    dynamics_model::NonlinearODEModel,
    process_noise_model::ProcessNoiseModel,
    t::Float64,
    Δt::Float64,
    x::Vector,
    P::Matrix;
    save_everystep::Bool = false,
    saveat = (),
)
    n = dynamics_model.n
    if length(x) != n || size(P) != (n, n)
        throw(
            DimensionMismatch(
                "state length is $(length(x)) and covariance is $(size(P)); model state dimension is $n",
            ),
        )
    end

    T = eltype(x)
    u0 = zeros(T, n * (n + 1))
    u0[1:n] .= x
    for j in 1:n
        u0[n + (j - 1) * n + j] = one(T)
    end

    tF = t + Δt
    prob = remake(
        dynamics_model.base_ode_augmented;
        u0 = u0,
        tspan = (t, tF),
        p = dynamics_model.p,
    )
    sol = solve(
        prob,
        dynamics_model.method;
        reltol = dynamics_model.reltol,
        abstol = dynamics_model.abstol,
        save_everystep = save_everystep,
        saveat = saveat,
    )
    if abs(sol.t[end] - tF) > 1e-8 * max(1.0, abs(tF))
        error("ODE propagation ended at t=$(sol.t[end]) instead of $tF (retcode=$(sol.retcode))")
    end

    uF = sol.u[end]
    x .= @view uF[1:n]
    Φ = reshape(@view(uF[(n + 1):end]), n, n)
    mapped = Φ * P * Φ' + get_noise(process_noise_model, Δt)
    P .= (mapped + mapped') ./ 2
    return tF, sol
end
