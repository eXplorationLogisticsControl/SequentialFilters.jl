using LinearAlgebra
using Random
using Test

using SequentialFilters

@testset "particle filter" begin
    rng = MersenneTwister(11)
    pf = ParticleFilter(0.0, [0.0], [1.0;;]; n_particles = 40, resampling = :systematic, rng = rng)
    @test size(pf.particles) == (1, 40)
    @test length(pf.weights) == 40
    @test sum(pf.weights) ≈ 1.0
    @test all(pf.weights .≈ 1 / 40)

    n0 = size(pf.particles, 2)
    resample!(pf)
    @test size(pf.particles, 2) == n0
    @test length(pf.weights) == n0
    @test sum(pf.weights) ≈ 1.0
    @test all(pf.weights .≈ 1 / n0)

    multi = ParticleFilter(
        0.0,
        [0.0, 0.0],
        Matrix{Float64}(I, 2, 2);
        n_particles = 25,
        resampling = :multinomial,
        rng = MersenneTwister(3),
    )
    resample!(multi)
    @test size(multi.particles, 2) == 25
    @test sum(multi.weights) ≈ 1.0
    @test_throws ArgumentError ParticleFilter(0.0, [0.0], [1.0;;]; resampling = :other)
    @test_throws DimensionMismatch ParticleFilter(0.0, [0.0, 0.0], [1.0;;])

    smoke = ParticleFilter(0.0, [0.0], [1.0;;]; n_particles = 400, rng = MersenneTwister(5))
    x_before = copy(smoke.x)
    dynamics = DiscreteDynamicsModel((t, x, dt) -> x; n = 1)
    sol = predict!(smoke, dynamics, AdditiveProcessNoise([0.25;;]), 1.0)
    @test size(smoke.particles, 2) == 400
    @test sum(smoke.weights) ≈ 1.0
    @test all(isfinite, smoke.particles)
    @test all(isfinite, smoke.x)
    @test all(isfinite, smoke.P)
    @test sol.t[end] ≈ 1.0
    @test length(sol.u[end]) == 2
    @test smoke.P[1, 1] > 0.2

    y = [3.0]
    R = [0.05;;]
    model = NonlinearMeasurementModel((t, x) -> ([x[1]], [1.0;;]))
    update!(smoke, model, y, R)
    @test size(smoke.particles, 2) == 400
    @test sum(smoke.weights) ≈ 1.0
    @test all(isfinite, smoke.x)
    @test all(isfinite, smoke.particles)
    @test abs(smoke.x[1] - y[1]) < abs(x_before[1] - y[1])

    function still!(du, u, p, t)
        du[1] = 0.0
        return nothing
    end
    cloud = ParticleFilter(0.0, [1.0], [0.01;;]; n_particles = 8, rng = MersenneTwister(9))
    ode = predict!(
        cloud,
        NonlinearODEModel(still!; n = 1),
        AdditiveProcessNoise(zeros(1, 1)),
        0.4;
        saveat = 0.0:0.2:0.4,
        ensemblealg = EnsembleSerial(),
    )
    @test ode.t[1] ≈ 0.0
    @test ode.t[end] ≈ 0.4
    @test length(ode.u[end]) == 2
    @test all(isfinite, ode.u[end])
    @test abs(cloud.x[1] - 1.0) < 0.2

    collapsed = ParticleFilter(0.0, [0.0], [1.0;;]; n_particles = 12, rng = MersenneTwister(4))
    update!(collapsed, (t, x) -> x, [20.0], [1.0e-16;;])
    @test size(collapsed.particles, 2) == 12
    sol_collapsed = predict!(
        collapsed,
        DiscreteDynamicsModel((t, x, dt) -> x; n = 1),
        AdditiveProcessNoise(zeros(1, 1)),
        1.0,
    )
    @test length(sol_collapsed.u[end]) == 2
    @test all(isfinite, sol_collapsed.u[end])
    @test all(isfinite, collapsed.x)
    @test all(isfinite, collapsed.P)
end
