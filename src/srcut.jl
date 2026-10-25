"""
    SquareRootConjugateUnscentedKalmanFilter(t, x, P; order=4)

Square-root conjugate unscented Kalman filter with additive noise.

Sigma points and weights come from `cut_points`. `order` is the CUT order
that generator already supports: `4` (CUT4, every `n ≥ 1`) or `6` (CUT6,
`2 ≤ n ≤ 9`). The default is `4`. CUT8 is not implemented.

The covariance is a lower Cholesky factor `S` with `P = S * S'` and a positive
diagonal, the same convention as `SquareRootUnscentedKalmanFilter`. The
constructor factors `P` once. Predict rebuilds `S` from the CUT weights with a
thin QR and, when a weight is negative, a Cholesky rank-one downdate. The
measurement update forms the innovation factor the same way, then rebuilds the
posterior factor with a Joseph QR of `[(S - K*C)'; (K*Sr)']`, where `C`
satisfies `S * C' = Pxy` and `Sr` is a factor of `Sy*Sy' - C*C'`. They do not
form `P` and call `cholesky` again. The element type follows `eltype(x)` and
`eltype(P)`, as in `ConjugateUnscentedKalmanFilter`.

CUT uses one weight for each sigma point, and that weight enters both the mean
and the covariance. A positive weight `w` contributes the column
`√w (χ − μ)` to the QR, together with a factor of the additive noise. A zero
weight is omitted. A negative weight is not placed in the QR; after the QR,
the factor is downdated by `√(−w) (χ − μ)` with the same rank-one Cholesky
downdate used by the square-root UKF (`σ = −1`). Supported CUT4 and CUT6
points have nonnegative weights, so those recursions only take the QR branch.
The downdate is the rule if a weight is negative. Several negative weights are
applied in column order, each to the factor left by the previous downdate.

No Jacobians are required.
"""
mutable struct SquareRootConjugateUnscentedKalmanFilter{T<:AbstractFloat}
    t::T
    x::Vector{T}
    S::Matrix{T}
    nx::Int
    order::Int

    function SquareRootConjugateUnscentedKalmanFilter(
        t::Real,
        x::AbstractVector{<:Real},
        P::AbstractMatrix{<:Real};
        order::Integer = 4,
    )
        nx = length(x)
        nx >= 1 || throw(ArgumentError("state dimension must be positive, got $nx"))
        if size(P) != (nx, nx)
            throw(DimensionMismatch("P is $(size(P)); expected ($nx, $nx)"))
        end
        order_int = Int(order)
        _check_cut_support(nx, order_int)
        T = _filter_eltype(x, P)
        S = Matrix(LowerTriangular(_lower_cholesky(Matrix{T}(P))))
        return new{T}(T(t), Vector{T}(x), S, nx, order_int)
    end
end


"""
    covariance(filter::SquareRootConjugateUnscentedKalmanFilter)

Reconstructed covariance `P = S * S'`, symmetrized.
"""
function covariance(filter::SquareRootConjugateUnscentedKalmanFilter)
    P = filter.S * filter.S'
    return (P + P') ./ 2
end


"""
    predict!(filter::SquareRootConjugateUnscentedKalmanFilter, dynamics_model::DynamicsModel, process_noise_model::ProcessNoiseModel, Δt::Float64; save_everystep=false, saveat=(), ensemblealg=EnsembleSerial())

Propagate `x + S ξ` with [`EnsemblePropagation`](@ref) and rebuild `S` from
the CUT weights and `Q = get_noise(process_noise_model, Δt)`.

The default `ensemblealg` is [`EnsembleSerial`](@ref). The returned
`ODESolution` stores `[μ; vec(Φ)]` for the noise-free cubature covariance.
"""
function predict!(
    filter::SquareRootConjugateUnscentedKalmanFilter,
    dynamics_model::DynamicsModel,
    process_noise_model::ProcessNoiseModel,
    Δt::Float64;
    save_everystep::Bool = false,
    saveat = (),
    ensemblealg = EnsembleSerial(),
)
    _check_step(Δt)
    T = eltype(filter.x)
    nx = filter.nx
    Qm = _as_eltype(_square_matrix(get_noise(process_noise_model, Δt), nx, "Q"), T)
    X, w = _srcut_sigma_points(filter)
    P_start = Matrix{Float64}(covariance(filter))
    Xprop, sol = _cloud_predict(
        dynamics_model,
        X,
        P_start,
        w,
        w,
        Float64(filter.t),
        Δt;
        ensemblealg = ensemblealg,
        save_everystep = save_everystep,
        saveat = saveat,
    )
    μ = _weighted_column_mean(Xprop, w)
    Snew = _srcut_factor(_column_deviations(Xprop, μ), w, _noise_factor(Qm))
    filter.x .= μ
    filter.S .= Snew
    filter.t = T(filter.t + Δt)
    return sol
end


"""
    update!(filter::SquareRootConjugateUnscentedKalmanFilter, h, y, R)

`h(t, x)` is the noise-free measurement model. `y` is the measurement and `R`
its additive covariance. The innovation factor uses the same QR and negative-weight
downdate as `predict!`. The gain is `Pxy (Sy Sy')⁻¹`, with `Pxy` the CUT
cross-covariance. The posterior factor is the Joseph form
`[(S - K*C)'; (K*Sr)']`, where `C` satisfies `S * C' = Pxy` and `Sr` is a
factor of `Sy*Sy' - C*C'`.
"""
function update!(filter::SquareRootConjugateUnscentedKalmanFilter, h, y, R)
    T = eltype(filter.x)
    yv = _measurement_vector(y, T)
    ny = length(yv)
    nx = filter.nx
    Rm = _as_eltype(_square_matrix(R, ny, "R"), T)
    X, w = _srcut_sigma_points(filter)
    Y = Matrix{T}(undef, ny, size(X, 2))
    for i in axes(X, 2)
        yi = h(filter.t, X[:, i])
        length(yi) == ny || throw(
            DimensionMismatch("h returned length $(length(yi)); expected $ny"),
        )
        for j in 1:ny
            Y[j, i] = yi[j]
        end
    end
    yhat = _weighted_column_mean(Y, w)
    Sy = _srcut_factor(_column_deviations(Y, yhat), w, _noise_factor(Rm))
    Pxy = _cross_covariance(X, filter.x, Y, yhat, w)
    Sl = LowerTriangular(Sy)
    K = (Pxy / Sl') / Sl
    xnew = filter.x .+ K * (yv .- yhat)
    C = (LowerTriangular(filter.S) \ Pxy)'
    IKS = filter.S - K * C
    Sr = _noise_factor(_symmetrize(Sy * Sy' - C * C'))
    Kr = size(Sr, 2) == 0 ? zeros(T, nx, 0) : K * Sr
    filter.S .= _qr_lower_factor(size(Kr, 2) == 0 ? IKS' : vcat(IKS', Kr'))
    filter.x .= xnew
    return nothing
end


function _srcut_sigma_points(filter::SquareRootConjugateUnscentedKalmanFilter{T}) where {T}
    ξ64, w64 = cut_points(filter.nx; order = filter.order)
    ξ = Matrix{T}(ξ64)
    w = Vector{T}(w64)
    X = filter.S * ξ .+ reshape(filter.x, :, 1)
    return X, w
end


"""
Square-root covariance from CUT deviations and an additive noise factor.

Positive weights enter the thin QR. Each negative weight is then a Cholesky
rank-one downdate, in column order. Zero weights are skipped.
"""
function _srcut_factor(dev::AbstractMatrix{T}, w::AbstractVector, noise::AbstractMatrix) where {T}
    n, m = size(dev)
    if length(w) != m
        throw(DimensionMismatch("expected $m weights, got $(length(w))"))
    end
    pieces = Matrix{T}[]
    negatives = Tuple{Int, T}[]
    for i in 1:m
        wi = T(w[i])
        if wi > 0
            push!(pieces, reshape(sqrt(wi) .* dev[:, i], n, 1))
        elseif wi < 0
            push!(negatives, (i, wi))
        elseif wi != 0
            throw(ArgumentError("CUT weight must be finite, got $wi"))
        end
    end
    if size(noise, 2) > 0
        push!(pieces, Matrix{T}(noise))
    end
    isempty(pieces) && throw(ArgumentError("square-root CUT factor has no positive columns"))
    S = _qr_lower_factor(reduce(hcat, pieces)')
    for (i, wi) in negatives
        _cholesky_rank1!(S, sqrt(-wi) .* dev[:, i], -1)
    end
    return S
end
