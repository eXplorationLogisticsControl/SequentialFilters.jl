using LinearAlgebra
using Random
using Test

using SequentialFilters

function spd_lower_factor(S)
    @test istril(S)
    @test all(>(0), diag(S))
    P = S * S'
    @test norm(P - P') < 1e-8 * max(1.0, norm(P))
    return (P + P') ./ 2
end

@testset "square-root cubature Kalman filter" begin
    n = 3
    ξ, w = cubature_points(n)
    @test size(ξ) == (n, 2n)
    @test length(w) == 2n
    @test all(w .≈ 1 / (2n))
    @test ξ[:, 1] ≈ [sqrt(n), 0, 0]
    @test ξ[:, n + 1] ≈ [-sqrt(n), 0, 0]
    @test_throws ArgumentError cubature_points(0)

    P0 = [1.0 0.2; 0.2 0.8]
    sr = SquareRootCubatureKalmanFilter(0.0, [1.0, -0.5], P0)
    @test sr.S * sr.S' ≈ P0 atol = 1e-12
    spd_lower_factor(sr.S)
    @test_throws DimensionMismatch SquareRootCubatureKalmanFilter(0.0, [0.0, 0.0], [1.0;;])

    # Third-degree cubature integrates the mean and covariance of a linear
    # map exactly. The square-root QR is the same Gram matrix, so on this
    # well-conditioned problem the filter matches the Kalman update to 1e-12.
    F = [1.0 0.1; 0.0 1.0]
    x0 = [1.0, -0.5]
    Q = 0.01 * Matrix{Float64}(I, 2, 2)
    H = [1.0 -0.2]
    R = fill(0.05, 1, 1)
    y = 0.7
    x_pred = F * x0
    P_pred = Symmetric(F * P0 * F' + Q)
    S = Symmetric(H * Matrix(P_pred) * H' + R)
    K = Matrix(P_pred) * H' / S
    x_upd = x_pred + vec(K * (y .- H * x_pred))
    P_upd = Symmetric(Matrix(P_pred) - K * S * K')
    filt = SquareRootCubatureKalmanFilter(0.0, x0, P0)
    predict!(filt, DiscreteDynamicsModel((t, x, dt) -> F * x; n = 2), AdditiveProcessNoise(Q), 0.2)
    @test filt.t ≈ 0.2
    @test filt.x ≈ x_pred atol = 1e-12
    @test covariance(filt) ≈ P_pred atol = 1e-12
    update!(filt, (t, x) -> H * x, y, R)
    @test filt.x ≈ x_upd atol = 1e-12
    @test covariance(filt) ≈ P_upd atol = 1e-12
    spd_lower_factor(filt.S)

    x32 = Float32[0.0, 1.0]
    P32 = Matrix{Float32}(I, 2, 2)
    sr32 = SquareRootCubatureKalmanFilter(0.0, x32, P32)
    @test eltype(sr32.S) == Float32
    predict!(
        sr32,
        DiscreteDynamicsModel((t, x, dt) -> Float32[1 0.2; 0 1] * x; n = 2),
        AdditiveProcessNoise(0.05f0 * Matrix{Float32}(I, 2, 2)),
        1.0,
    )
    @test eltype(sr32.x) == Float32
    @test all(isfinite, sr32.x)
    spd_lower_factor(Matrix{Float64}(sr32.S))
end
