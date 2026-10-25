using LinearAlgebra
using Random
using Test

using SequentialFilters

@testset "diagnosis" begin
    @test estimation_error([5.0, 1.0], [2.0, -1.0]) ≈ [3.0, 2.0]

    errors = [3.0 0.0; 4.0 0.0]
    @test rmse(errors) ≈ sqrt(12.5)
    @test rmse([3.0, 4.0]) ≈ 5.0

    blocked = [3.0 0.0; 4.0 0.0; 0.0 12.0]
    @test block_rmse(blocked, 1:2) ≈ sqrt(12.5)
    @test block_rmse(blocked, 3:3) ≈ sqrt(72.0)
    parts = block_rmse(blocked, (position = 1:2, velocity = 3:3))
    @test parts.position ≈ sqrt(12.5)
    @test parts.velocity ≈ sqrt(72.0)

    @test time_averaged_rmse([2.0, 4.0]) ≈ 3.0
    history = cat([3.0 0.0; 4.0 0.0], [0.0 0.0; 0.0 6.0]; dims = 3)
    @test rmse(history) ≈ [sqrt(12.5), sqrt(18.0)]
    @test time_averaged_rmse(history) ≈ (sqrt(12.5) + sqrt(18.0)) / 2
    averaged_blocks = time_averaged_rmse(history, (position = 1:2, velocity = 1:1))
    @test averaged_blocks.position ≈ (sqrt(12.5) + sqrt(18.0)) / 2

    @test nees([1.0, 0.0], Matrix{Float64}(I, 2, 2)) ≈ 1.0
    @test nees([2.0, 0.0], [1.0, 0.0], Matrix{Float64}(I, 2, 2)) ≈ 1.0
    @test averaged_nees([1.0, 3.0, NaN]) ≈ 2.0

    rng = MersenneTwister(1)
    covariance = Diagonal([1.0, 4.0, 0.25, 9.0])
    factor = Diagonal(sqrt.(diag(covariance)))
    samples = [nees(factor * randn(rng, 4), covariance) for _ in 1:20_000]
    @test averaged_nees(samples) ≈ 4 atol = 0.1

    @test nis([1.0, 0.0], Matrix{Float64}(I, 2, 2)) ≈ 1.0
    @test nis([2.0, 0.0], Diagonal([4.0, 1.0])) ≈ 1.0

    series = [1.0 0.0 1.0; 0.0 1.0 0.0]
    rho = innovation_autocorrelation(series)
    @test rho[1] ≈ 1.0
    @test innovation_autocorrelation([1.0, 1.0, 1.0]) ≈ [1.0, 2 / 3, 1 / 3]
    @test whiteness_bound(size(series, 2)) ≈ 1.96 / sqrt(size(series, 2))

    @test is_diverged([10.0, 0.0]; threshold = 5)
    @test is_diverged(10.0; threshold = 5)
    @test !is_diverged([1.0, 0.0]; threshold = 5)
    @test is_diverged([1.0, 0.0]; threshold = 5, estimate = [NaN, 0.0])
    @test is_diverged(0.0; threshold = 5, covariance = [1.0 NaN])
    @test divergence_rate([true, false, false, false]) ≈ 0.25

    @test mahalanobis([3.0, 0.0], [0.0, 0.0], I) ≈ 9.0
    @test mahalanobis([3.0, 0.0], [0.0, 0.0], I; square_root = true) ≈ 3.0
end
