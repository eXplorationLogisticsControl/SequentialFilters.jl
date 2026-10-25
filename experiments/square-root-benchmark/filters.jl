"""
Registry of every filter in SequentialFilters.

Each entry builds a filter from `(t, x, P)`, predicts, reports the innovation
and its predicted covariance, then updates. Square-root filters expose `P`
through `covariance`. Sigma-point innovation covariances use the public
sigma-point and weight functions, with the same weights the filter uses for
its measurement update.
"""

struct FilterSpec
    name::String
    build::Function
    predict::Function
    innovation::Function
    update::Function
    mean::Function
    covariance::Function
end

function _weighted_mean(Y, w)
    μ = zeros(Float64, size(Y, 1))
    for i in axes(Y, 2)
        μ .+= w[i] .* view(Y, :, i)
    end
    return μ
end

function _weighted_covariance(Y, μ, w)
    ny = size(Y, 1)
    S = zeros(Float64, ny, ny)
    for i in axes(Y, 2)
        d = Y[:, i] .- μ
        S .+= w[i] .* (d * d')
    end
    return (S + S') ./ 2
end

function _mapped_innovation(X, w_mean, w_cov, h, t, y, R)
    ny = length(y)
    Y = Matrix{Float64}(undef, ny, size(X, 2))
    for i in axes(X, 2)
        Y[:, i] = h(t, X[:, i])
    end
    yhat = _weighted_mean(Y, w_mean)
    S = _weighted_covariance(Y, yhat, w_cov) + Matrix{Float64}(R)
    S = (S + S') ./ 2
    return collect(Float64, y) .- yhat, S
end

function _linear_innovation(x, P, h, H, t, y, R)
    yhat = h(t, x)
    Hm = Matrix{Float64}(H(t, x))
    S = Hm * Matrix{Float64}(P) * Hm' + Matrix{Float64}(R)
    return collect(Float64, y) .- collect(Float64, yhat), (S + S') ./ 2
end

function _unscented_innovation(x, P, h, t, y, R; alpha, beta, kappa)
    X = SequentialFilters.get_sigma_points(x, P; alpha = alpha, beta = beta, kappa = kappa)
    weights = SequentialFilters.unscented_weights(length(x); alpha = alpha, beta = beta, kappa = kappa)
    return _mapped_innovation(X, weights.Wm, weights.Wc, h, t, y, R)
end

function _cut_innovation(x, P, h, t, y, R; order)
    ξ, w = SequentialFilters.cut_points(length(x); order = order)
    factor = Matrix(cholesky(Hermitian((Matrix{Float64}(P) + Matrix{Float64}(P)') ./ 2)).L)
    X = factor * ξ .+ reshape(collect(Float64, x), :, 1)
    weights = collect(Float64, w)
    return _mapped_innovation(X, weights, weights, h, t, y, R)
end

"""EKF, UKF, square-root EKF, square-root UKF, CUT, and square-root CUT."""
function package_filters()
    SF = SequentialFilters
    return [
        FilterSpec(
            "EKF",
            (t, x, P) -> SF.ExtendedKalmanFilter(Float64(t), Vector{Float64}(x), Matrix{Float64}(P)),
            (filt, scenario) -> SF.predict!(filt, scenario.dynamics, scenario.process_noise, Float64(scenario.Δt)),
            (filt, scenario, y) -> _linear_innovation(filt.x, filt.P, scenario.h, scenario.H, filt.t, y, scenario.R),
            (filt, scenario, y) -> SF.update!(filt, scenario.measurements, y, scenario.R),
            filt -> filt.x,
            filt -> filt.P,
        ),
        FilterSpec(
            "UKF",
            (t, x, P) -> SF.UnscentedKalmanFilter(t, x, P),
            (filt, scenario) -> SF.predict!(filt, scenario.f, scenario.Q, scenario.Δt),
            (filt, scenario, y) -> _unscented_innovation(
                filt.x, filt.P, scenario.h, filt.t, y, scenario.R;
                alpha = filt.alpha, beta = filt.beta, kappa = filt.kappa,
            ),
            (filt, scenario, y) -> SF.update!(filt, scenario.h, y, scenario.R),
            filt -> filt.x,
            filt -> filt.P,
        ),
        FilterSpec(
            "SR-EKF",
            (t, x, P) -> SF.SquareRootExtendedKalmanFilter(t, x, P),
            (filt, scenario) -> SF.predict!(filt, scenario.f_variational, scenario.F, scenario.Q, scenario.Δt),
            (filt, scenario, y) -> _linear_innovation(
                filt.x, SF.covariance(filt), scenario.h, scenario.H, filt.t, y, scenario.R,
            ),
            (filt, scenario, y) -> SF.update!(filt, scenario.h, scenario.H, y, scenario.R),
            filt -> filt.x,
            filt -> SF.covariance(filt),
        ),
        FilterSpec(
            "SR-UKF",
            (t, x, P) -> SF.SquareRootUnscentedKalmanFilter(t, x, P),
            (filt, scenario) -> SF.predict!(filt, scenario.f, scenario.Q, scenario.Δt),
            (filt, scenario, y) -> _unscented_innovation(
                filt.x, SF.covariance(filt), scenario.h, filt.t, y, scenario.R;
                alpha = filt.alpha, beta = filt.beta, kappa = filt.kappa,
            ),
            (filt, scenario, y) -> SF.update!(filt, scenario.h, y, scenario.R),
            filt -> filt.x,
            filt -> SF.covariance(filt),
        ),
        FilterSpec(
            "CUT",
            (t, x, P) -> SF.ConjugateUnscentedKalmanFilter(t, x, P; order = 4),
            (filt, scenario) -> SF.predict!(filt, scenario.f, scenario.Q, scenario.Δt),
            (filt, scenario, y) -> _cut_innovation(
                filt.x, filt.P, scenario.h, filt.t, y, scenario.R; order = filt.order,
            ),
            (filt, scenario, y) -> SF.update!(filt, scenario.h, y, scenario.R),
            filt -> filt.x,
            filt -> filt.P,
        ),
        FilterSpec(
            "SR-CUT",
            (t, x, P) -> SF.SquareRootConjugateUnscentedKalmanFilter(t, x, P; order = 4),
            (filt, scenario) -> SF.predict!(filt, scenario.f, scenario.Q, scenario.Δt),
            (filt, scenario, y) -> _cut_innovation(
                filt.x, SF.covariance(filt), scenario.h, filt.t, y, scenario.R; order = filt.order,
            ),
            (filt, scenario, y) -> SF.update!(filt, scenario.h, y, scenario.R),
            filt -> filt.x,
            filt -> SF.covariance(filt),
        ),
    ]
end
