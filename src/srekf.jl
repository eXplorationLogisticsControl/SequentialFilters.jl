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
    predict!(filter::SquareRootExtendedKalmanFilter, f, F, Q, Δt)

Noise-free process model `f(t, x, Δt)` and its Jacobian. `F` is either an
`nx × nx` matrix or a function `F(t, x, Δt)` returning that Jacobian. `Q` is
the additive process-noise covariance.

The predicted factor comes from the thin QR of `[(F*S)'; Sq']`, where `Sq`
is a factor of `Q` (`Q = Sq * Sq'`). The stored `S` is the lower-triangular
transpose of that R factor, with a positive diagonal.
"""
function predict!(filter::SquareRootExtendedKalmanFilter, f, Fjac, Q, Δt::Real)
    nx = filter.nx
    Qm = _square_matrix(Q, nx, "Q")
    Δt64 = Float64(Δt)
    isfinite(Δt64) || throw(ArgumentError("Δt must be finite, got $Δt"))
    t = filter.t
    xnew = f(t, filter.x, Δt64)
    length(xnew) == nx || throw(
        DimensionMismatch("f returned length $(length(xnew)); expected $nx"),
    )
    F = _state_jacobian(Fjac, t, filter.x, Δt64)
    if size(F) != (nx, nx)
        throw(DimensionMismatch("F is $(size(F)); expected ($nx, $nx)"))
    end
    FS = F * filter.S
    Nf = _noise_factor(Qm)
    M = size(Nf, 2) == 0 ? FS' : vcat(FS', Nf')
    filter.S .= _qr_lower_factor(M)
    for j in 1:nx
        filter.x[j] = xnew[j]
    end
    filter.t = t + Δt64
    return nothing
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
    n = size(M, 1)
    if norm(M) == 0
        return zeros(n, 0)
    end
    factorization = cholesky(Hermitian(M); check = false)
    if issuccess(factorization)
        return Matrix(factorization.L)
    end
    ev = eigen(Symmetric(M))
    scale = max(1.0, maximum(abs, ev.values))
    tol = 1e-12 * scale
    if any(<( -tol ), ev.values)
        throw(ArgumentError("noise covariance is not positive semidefinite"))
    end
    vals = max.(ev.values, 0.0)
    keep = vals .> tol
    if !any(keep)
        return zeros(n, 0)
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
