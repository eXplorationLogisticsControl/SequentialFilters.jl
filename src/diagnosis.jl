"""
Elementary filter diagnosis.

Estimation errors are truth minus estimate. Root-mean-square error at one
time, over runs whose errors are finite, is

```
RMSE = sqrt(mean_i ||x - x̂||²)
```

NEES, NIS, and the Mahalanobis distance are the quadratic form `e' * (A \\ e)`
on a symmetrized covariance. The solve is `A \\ e`, not an explicit inverse.
Innovation whiteness is the sample autocorrelation of a vector innovation
sequence. A run is diverged when an error exceeds a caller-supplied threshold
or an estimate or covariance is not finite.
"""

function _vector(x)
    return collect(Float64, vec(x))
end

function _square(A, n::Int)
    if A isa UniformScaling
        return Matrix{Float64}(A, n, n)
    end
    M = Matrix{Float64}(A)
    if size(M) != (n, n)
        throw(DimensionMismatch("covariance is $(size(M)); expected ($n, $n)"))
    end
    return M
end

function _diagnosis_symmetrize(A::AbstractMatrix)
    return (A + A') ./ 2
end

"""
    estimation_error(x, xhat)

Truth minus estimate, `x - x̂`, as a `Vector{Float64}`.
"""
function estimation_error(x, xhat)
    xv = _vector(x)
    hv = _vector(xhat)
    if length(xv) != length(hv)
        throw(
            DimensionMismatch(
                "truth length is $(length(xv)); estimate length is $(length(hv))",
            ),
        )
    end
    return xv .- hv
end

"""
    _quadratic_form(e, A)

`e' * (A \\ e)` after replacing `A` with `(A + A') / 2`.
"""
function _quadratic_form(e, A)
    ev = _vector(e)
    M = _diagnosis_symmetrize(_square(A, length(ev)))
    return dot(ev, Symmetric(M) \ ev)
end

function _state_index(indices)
    return indices isa Integer ? (indices:indices) : indices
end

"""
    rmse(errors)

Root-mean-square estimation error.

A vector is one error sample, and the result is its Euclidean norm. A matrix
is `state × run`: each column is one run, and non-finite columns are omitted.
A `state × run × time` array returns one RMSE per time.
"""
function rmse(error::AbstractVector{<:Real})
    if !all(isfinite, error)
        return NaN
    end
    return sqrt(sum(abs2, error))
end

function rmse(errors::AbstractMatrix{<:Real})
    acc = 0.0
    nfin = 0
    for i in 1:size(errors, 2)
        col = view(errors, :, i)
        if all(isfinite, col)
            acc += sum(abs2, col)
            nfin += 1
        end
    end
    return nfin == 0 ? NaN : sqrt(acc / nfin)
end

function rmse(errors::AbstractArray{<:Real,3})
    return [rmse(@view errors[:, :, k]) for k in 1:size(errors, 3)]
end

function _block_view(errors::AbstractMatrix, indices)
    return view(errors, _state_index(indices), :)
end

function _block_view(errors::AbstractArray{<:Real,3}, indices)
    return view(errors, _state_index(indices), :, :)
end

"""
    block_rmse(errors, indices)
    block_rmse(errors, blocks::NamedTuple)

RMSE on a state block. `indices` selects rows. A named tuple such as
`(position=1:3, velocity=4:6)` returns a named tuple of block RMSEs.
"""
function block_rmse(errors::AbstractArray{<:Real}, indices)
    return rmse(_block_view(errors, indices))
end

function block_rmse(errors::AbstractArray{<:Real}, blocks::NamedTuple)
    values = ntuple(i -> block_rmse(errors, blocks[i]), length(blocks))
    return NamedTuple{keys(blocks)}(values)
end

"""
    time_averaged_rmse(rmse_history)
    time_averaged_rmse(errors::AbstractArray{<:Real,3})
    time_averaged_rmse(errors, indices_or_blocks)

Mean of a RMSE history. Non-finite samples are omitted. For a
`state × run × time` array this is the mean of `rmse` over time. Pass
`errors[:, :, posterior]` to average measurement instants only.
"""
function time_averaged_rmse(rmse_history::AbstractVector{<:Real})
    acc = 0.0
    nfin = 0
    for value in rmse_history
        if isfinite(value)
            acc += value
            nfin += 1
        end
    end
    return nfin == 0 ? NaN : acc / nfin
end

function time_averaged_rmse(errors::AbstractArray{<:Real,3})
    return time_averaged_rmse(rmse(errors))
end

function time_averaged_rmse(errors::AbstractArray{<:Real,3}, indices)
    return time_averaged_rmse(block_rmse(errors, indices))
end

function time_averaged_rmse(errors::AbstractArray{<:Real,3}, blocks::NamedTuple)
    values = ntuple(i -> time_averaged_rmse(errors, blocks[i]), length(blocks))
    return NamedTuple{keys(blocks)}(values)
end

"""
    nees(error, P)
    nees(x, xhat, P)

Normalized estimation error squared, `(x - x̂)' P⁻¹ (x - x̂)`, computed as
`e' * (P \\ e)` on the symmetrized covariance.
"""
function nees(error::AbstractVector, P)
    return _quadratic_form(error, P)
end

function nees(x, xhat, P)
    return nees(estimation_error(x, xhat), P)
end

"""
    averaged_nees(values)

Mean NEES. A vector is averaged directly. A matrix is averaged down each
column (one column per time). Non-finite values are omitted.
"""
function averaged_nees(values::AbstractVector)
    acc = 0.0
    nfin = 0
    for value in values
        if isfinite(value)
            acc += value
            nfin += 1
        end
    end
    return nfin == 0 ? NaN : acc / nfin
end

function averaged_nees(values::AbstractMatrix)
    return [averaged_nees(@view values[:, j]) for j in 1:size(values, 2)]
end

"""
    nis(innovation, S)

Normalized innovation squared, `ν' S⁻¹ ν`, with the same symmetrized solve as
[`nees`](@ref). `S` is the innovation covariance supplied by the caller.
"""
function nis(innovation, S)
    return _quadratic_form(innovation, S)
end

"""
    averaged_nis(values)

Mean NIS. A vector is one time. A matrix is averaged down each column.
Non-finite values are omitted.
"""
averaged_nis(values::AbstractVector) = averaged_nees(values)
averaged_nis(values::AbstractMatrix) = averaged_nees(values)

"""
    innovation_autocorrelation(nu; max_lag=nothing)

Sample autocorrelation of a vector innovation sequence. `nu` is `ny × N`
(time along columns) or a vector (one measurement channel).

For lag `ℓ`,

```
ρ(ℓ) = Σ_{k=1}^{N-ℓ} ν_k · ν_{k+ℓ}  /  Σ_{k=1}^{N} ||ν_k||²
```

Lag 0 is exactly 1 when the denominator is positive. Returns lags `0, 1, …`
in a vector whose first entry is lag 0.
"""
function innovation_autocorrelation(nu::AbstractVector; max_lag::Union{Nothing,Integer} = nothing)
    return innovation_autocorrelation(reshape(_vector(nu), 1, :); max_lag = max_lag)
end

function innovation_autocorrelation(nu::AbstractMatrix; max_lag::Union{Nothing,Integer} = nothing)
    series = Matrix{Float64}(nu)
    N = size(series, 2)
    N >= 1 || throw(ArgumentError("innovation sequence must contain at least one sample"))
    last = max_lag === nothing ? N - 1 : min(N - 1, Int(max_lag))
    last >= 0 || throw(ArgumentError("max_lag must be nonnegative, got $max_lag"))
    denom = sum(abs2, series)
    ρ = Vector{Float64}(undef, last + 1)
    for ell in 0:last
        acc = 0.0
        for k in 1:(N - ell)
            acc += dot(view(series, :, k), view(series, :, k + ell))
        end
        ρ[ell + 1] = acc / denom
    end
    return ρ
end

"""
    whiteness_bound(N)

Usual normal bound `1.96 / sqrt(N)` for the autocorrelation of a white
innovation sequence at a nonzero lag. `N` is the number of innovations.
"""
function whiteness_bound(N::Integer)
    n = Int(N)
    n >= 1 || throw(ArgumentError("N must be positive, got $N"))
    return 1.96 / sqrt(n)
end

"""
    is_diverged(error; threshold, estimate=nothing, covariance=nothing)

Whether one run has diverged. A real `error` is already a magnitude. A vector
`error` uses its Euclidean norm. The run diverges when that magnitude exceeds
`threshold`, or when `error`, `estimate`, or `covariance` is not finite.
"""
function is_diverged(error; threshold::Real, estimate = nothing, covariance = nothing)
    if estimate !== nothing && any(!isfinite, estimate)
        return true
    end
    if covariance !== nothing && any(!isfinite, covariance)
        return true
    end
    magnitude = error isa Real ? abs(Float64(error)) : sqrt(sum(abs2, _vector(error)))
    if !isfinite(magnitude)
        return true
    end
    return magnitude > Float64(threshold)
end

"""
    divergence_rate(flags)

Fraction of runs that diverged.
"""
function divergence_rate(flags)
    n = length(flags)
    n == 0 && return NaN
    return count(identity, flags) / n
end

"""
    mahalanobis(x, xhat, P; square_root=false)

Mahalanobis distance of the truth in the filter covariance. This is the same
quadratic form as [`nees`](@ref). The default returns that squared distance.
`square_root=true` returns its square root.
"""
function mahalanobis(x, xhat, P; square_root::Bool = false)
    squared = nees(x, xhat, P)
    return square_root ? sqrt(squared) : squared
end
