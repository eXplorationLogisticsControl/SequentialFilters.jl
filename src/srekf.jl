"""
    SquareRootExtendedKalmanFilter(t, x, P)

Discrete extended Kalman filter that stores the covariance as a lower
Cholesky factor `S` with `P = S * S'` and a positive diagonal.

`S` is lower triangular. The constructor factors `P` once. Later predict and
update steps rebuild `S` with a QR factorization (equivalent to a stable
Cholesky update). They do not form `P` and call `cholesky` again.

Process and measurement noise are additive. `f` and `h` are nonlinear.
Jacobians are supplied by the caller, either as matrices or as functions.
"""
mutable struct SquareRootExtendedKalmanFilter
    t::Float64
    x::Vector{Float64}
    S::Matrix{Float64}
    nx::Int

    function SquareRootExtendedKalmanFilter(
        t::Real,
        x::AbstractVector{<:Real},
        P::AbstractMatrix{<:Real},
    )
        nx = length(x)
        nx >= 1 || throw(ArgumentError("state dimension must be positive, got $nx"))
        if size(P) != (nx, nx)
            throw(DimensionMismatch("P is $(size(P)); expected ($nx, $nx)"))
        end
        S = Matrix(LowerTriangular(_lower_cholesky(P)))
        return new(Float64(t), Vector{Float64}(x), S, nx)
    end
end


"""
    covariance(filter::SquareRootExtendedKalmanFilter)

Reconstructed covariance `P = S * S'`, symmetrized.
"""
function covariance(filter::SquareRootExtendedKalmanFilter)
    P = filter.S * filter.S'
    return (P + P') ./ 2
end


"""
    predict!(filter::SquareRootExtendedKalmanFilter, dynamics_model::DynamicsModel, process_noise_model::ProcessNoiseModel, Δt::Float64; save_everystep=false, saveat=())

Propagate the mean with `dynamics_model` and rebuild the covariance factor
from the state-transition matrix.

A [`NonlinearODEModel`](@ref) integrates the same augmented state as the
extended Kalman filter. `Φ` is read from that `ODESolution`, which is
returned. A [`DiscreteDynamicsModel`](@ref) evaluates `f` and its Jacobian
(`discrete_jacobian`: the stored Jacobian, or ForwardDiff). The returned
solution still stores `[x; vec(Φ)]` so the EKF covariance plot can read it.

`Q = get_noise(process_noise_model, Δt)`. The predicted factor is the thin QR
of `[(Φ*S)'; Sq']`. This filter propagates one trajectory, so it has no
`ensemblealg` argument.
"""
function predict!(
    filter::SquareRootExtendedKalmanFilter,
    dynamics_model::DynamicsModel,
    process_noise_model::ProcessNoiseModel,
    Δt::Float64;
    save_everystep::Bool = false,
    saveat = (),
)
    _check_step(Δt)
    nx = filter.nx
    Qm = _square_matrix(get_noise(process_noise_model, Δt), nx, "Q")
    xnew, Φ, sol = _srekf_transition(
        filter,
        dynamics_model,
        process_noise_model,
        Δt;
        save_everystep = save_everystep,
        saveat = saveat,
    )
    if size(Φ) != (nx, nx)
        throw(DimensionMismatch("F is $(size(Φ)); expected ($nx, $nx)"))
    end
    length(xnew) == nx || throw(
        DimensionMismatch("dynamics returned length $(length(xnew)); expected $nx"),
    )
    FS = Φ * filter.S
    Nf = _noise_factor(Qm)
    M = size(Nf, 2) == 0 ? FS' : vcat(FS', Nf')
    filter.S .= _qr_lower_factor(M)
    for j in 1:nx
        filter.x[j] = xnew[j]
    end
    filter.t = filter.t + Δt
    return sol
end

function _srekf_transition(
    filter::SquareRootExtendedKalmanFilter,
    dynamics::NonlinearODEModel,
    noise::ProcessNoiseModel,
    Δt::Float64;
    save_everystep::Bool = false,
    saveat = (),
)
    nx = filter.nx
    x = copy(filter.x)
    P = Matrix{Float64}(I, nx, nx)
    tF, sol = propagate!(
        dynamics,
        noise,
        filter.t,
        Δt,
        x,
        P;
        save_everystep = save_everystep,
        saveat = saveat,
    )
    Φ = Matrix{Float64}(reshape(sol.u[end][(nx + 1):end], nx, nx))
    tF == filter.t + Δt || error("square-root EKF propagation ended at $tF")
    return x, Φ, sol
end

function _srekf_transition(
    filter::SquareRootExtendedKalmanFilter,
    dynamics::DiscreteDynamicsModel,
    ::ProcessNoiseModel,
    Δt::Float64;
    save_everystep::Bool = false,
    saveat = (),
)
    nx = filter.nx
    if dynamics.n != nx
        throw(DimensionMismatch("dynamics dimension is $(dynamics.n); filter state is $nx"))
    end
    t = filter.t
    xnew = dynamics.f(t, filter.x, Δt)
    length(xnew) == nx || throw(
        DimensionMismatch("f returned length $(length(xnew)); expected $nx"),
    )
    Φ = Matrix{Float64}(discrete_jacobian(dynamics, t, filter.x, Δt))
    if size(Φ) != (nx, nx)
        throw(DimensionMismatch("F is $(size(Φ)); expected ($nx, $nx)"))
    end
    sol = _discrete_transition_solution(filter.x, xnew, Φ, t, Δt; save_everystep, saveat)
    return collect(Float64, xnew), Φ, sol
end

function _discrete_transition_solution(x0, x1, Φ, t, Δt; save_everystep::Bool = false, saveat = ())
    times = _discrete_times(t, Δt, saveat, save_everystep)
    n = length(x0)
    x0v = collect(Float64, x0)
    x1v = collect(Float64, x1)
    Φ64 = Matrix{Float64}(Φ)
    eye = Matrix{Float64}(I, n, n)
    span = Δt == 0 ? 1.0 : Δt
    us = Vector{Vector{Float64}}(undef, length(times))
    for (k, τ) in enumerate(times)
        α = (τ - t) / span
        μ = (1 - α) .* x0v .+ α .* x1v
        Φk = (1 - α) .* eye .+ α .* Φ64
        u = zeros(Float64, n * (n + 1))
        u[1:n] .= μ
        u[(n + 1):end] .= vec(Φk)
        us[k] = u
    end
    prob = ODEProblem((du, u, p, t_ode) -> (du .= 0), us[1], (times[1], times[end]))
    return SciMLBase.build_solution(
        prob,
        Tsit5(),
        times,
        us;
        retcode = SciMLBase.ReturnCode.Success,
    )
end


"""
    update!(filter::SquareRootExtendedKalmanFilter, h, H, y, R)

Noise-free measurement model `h(t, x)` and its Jacobian. `H` is either an
`ny × nx` matrix or a function `H(t, x)`. `y` is the measurement and `R` its
additive covariance.

The innovation factor and the posterior factor are both thin QR factorizations.
The posterior QR is the Joseph form `[(I - K*H) S; (K*Sr)']`.
"""
function update!(filter::SquareRootExtendedKalmanFilter, h, Hjac, y, R)
    yhat = h(filter.t, filter.x)
    H = _measurement_jacobian(Hjac, filter.t, filter.x)
    return _srekf_update!(filter, yhat, H, y, R)
end

function update!(
    filter::SquareRootExtendedKalmanFilter,
    model::MeasurementModel,
    y,
    R,
)
    yhat, H = model.get_h_H(filter.t, filter.x)
    return _srekf_update!(filter, yhat, H, y, R)
end


function _srekf_update!(filter::SquareRootExtendedKalmanFilter, yhat, H, y, R)
    yv = _measurement_vector(y)
    ny = length(yv)
    yhatv = collect(Float64, yhat)
    length(yhatv) == ny || throw(
        DimensionMismatch("h returned length $(length(yhatv)); expected $ny"),
    )
    nx = filter.nx
    Hm = Matrix{Float64}(H)
    if size(Hm) != (ny, nx)
        throw(DimensionMismatch("H is $(size(Hm)); expected ($ny, $nx)"))
    end
    Rm = _square_matrix(R, ny, "R")
    S = filter.S
    HS = Hm * S
    Sr = _noise_factor(Rm)
    Sy = _qr_lower_factor(size(Sr, 2) == 0 ? HS' : vcat(HS', Sr'))
    Sl = LowerTriangular(Sy)
    PHT = S * HS'
    K = (PHT / Sl') / Sl
    xnew = filter.x .+ K * (yv .- yhatv)
    IKS = S - K * HS
    Kr = size(Sr, 2) == 0 ? zeros(nx, 0) : K * Sr
    posterior = size(Kr, 2) == 0 ? IKS' : vcat(IKS', Kr')
    filter.S .= _qr_lower_factor(posterior)
    filter.x .= xnew
    return nothing
end


"""
Lower Cholesky factor of a Gram matrix `M' * M`.

`M` is `m × n` with `m ≥ n`. The result is `n × n`, lower triangular, with a
nonnegative diagonal, and satisfies `S * S' = M' * M`.
"""
function _qr_lower_factor(M::AbstractMatrix)
    n = size(M, 2)
    if size(M, 1) < n
        throw(ArgumentError("QR covariance factor needs at least $n rows, got $(size(M, 1))"))
    end
    R = Matrix(qr(M).R)
    Ru = R[1:n, 1:n]
    for i in 1:n
        if Ru[i, i] < 0
            @views Ru[i, :] .*= -1
        end
    end
    return Matrix(LowerTriangular(Ru'))
end


"""
Matrix `N` such that `N * N'` equals the symmetrized noise covariance.

Positive definite inputs use the lower Cholesky factor. Positive semidefinite
inputs, including zero, use an eigenvalue factor and may have fewer than `n`
columns.
"""
function _noise_factor(A::AbstractMatrix)
    M = _symmetrize(A)
    T = eltype(M)
    n = size(M, 1)
    if norm(M) == 0
        return zeros(T, n, 0)
    end
    factorization = cholesky(Hermitian(M); check = false)
    if issuccess(factorization)
        return Matrix{T}(factorization.L)
    end
    ev = eigen(Symmetric(M))
    scale = max(one(T), T(maximum(abs, ev.values)))
    # Float64 keeps the historical 1e-12 relative cutoff. Lower-precision
    # values need a cutoff above their own roundoff, so the floor is 100 ulps.
    tol = max(T(1e-12), 100 * eps(T)) * scale
    if any(<( -tol ), ev.values)
        throw(ArgumentError("noise covariance is not positive semidefinite"))
    end
    vals = max.(ev.values, zero(T))
    keep = vals .> tol
    if !any(keep)
        return zeros(T, n, 0)
    end
    return ev.vectors[:, keep] * Diagonal(sqrt.(vals[keep]))
end

function _state_jacobian(J, t, x, Δt)
    if J isa AbstractMatrix
        return Matrix{Float64}(J)
    end
    return Matrix{Float64}(J(t, x, Δt))
end

function _measurement_jacobian(J, t, x)
    if J isa AbstractMatrix
        return Matrix{Float64}(J)
    end
    return Matrix{Float64}(J(t, x))
end
