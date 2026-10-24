using LinearAlgebra
using OrdinaryDiffEq
using Random
using Test

using SequentialFilters

function _cut_moment(X, w, f)
    s = 0.0
    for i in axes(X, 2)
        s += w[i] * f(@view X[:, i])
    end
    return s
end

function _nnz_counts(X)
    counts = Dict{Int, Int}()
    for i in axes(X, 2)
        k = count(!iszero, @view X[:, i])
        counts[k] = get(counts, k, 0) + 1
    end
    return counts
end

@testset "CUT" begin

@testset "point sets and Gaussian moments" begin
    cut4_counts = Dict(1 => 4, 2 => 9, 3 => 14, 6 => 76, 10 => 1044)
    for (n, N) in cut4_counts
        X, w = cut_points(n; order = 4)
        @test size(X) == (n, N)
        @test length(w) == N
        @test all(>=(0), w)
        @test sum(w) ≈ 1 atol = 1e-12
        @test sum(abs, w) ≈ 1 atol = 1e-12
        for i in 1:n
            @test abs(_cut_moment(X, w, x -> x[i])) < 1e-12
            @test _cut_moment(X, w, x -> x[i]^2) ≈ 1 atol = 1e-12
            @test _cut_moment(X, w, x -> x[i]^4) ≈ 3 atol = 1e-12
        end
        if n >= 2
            @test abs(_cut_moment(X, w, x -> x[1] * x[2])) < 1e-12
            @test _cut_moment(X, w, x -> x[1]^2 * x[2]^2) ≈ 1 atol = 1e-12
        end
    end

    X1, w1 = cut_points(1; order = 4)
    @test X1[1, :] ≈ [
        0.7419637843027252,
        -0.7419637843027252,
        2.3344142183389796,
        -2.3344142183389796,
    ]
    @test w1 ≈ [
        0.45412414523193145,
        0.45412414523193145,
        0.04587585476806854,
        0.04587585476806854,
    ]

    X3, = cut_points(3; order = 4)
    r1 = sqrt((3 + 2) / 2)
    r2 = sqrt((3 + 2) / (3 - 2))
    @test X3[:, 1] ≈ [r1, 0.0, 0.0]
    @test X3[:, 2] ≈ [-r1, 0.0, 0.0]
    @test X3[:, 7] ≈ [-r2, -r2, -r2]
    counts3 = _nnz_counts(X3)
    @test counts3[1] == 6
    @test counts3[3] == 8
    @test !haskey(counts3, 0)

    counts6 = _nnz_counts(cut_points(6; order = 4)[1])
    @test counts6[1] == 12
    @test counts6[6] == 64

    cut6_counts = Dict(2 => 13, 4 => 49, 6 => 137, 7 => 423, 9 => 1203)
    for (n, N) in cut6_counts
        X, w = cut_points(n; order = 6)
        @test size(X) == (n, N)
        @test length(w) == N
        @test all(>(0), w)
        @test sum(w) ≈ 1 atol = 1e-12
        @test sum(abs, w) ≈ 1 atol = 1e-12
        @test all(iszero, X[:, 1])
        for i in 1:n
            @test abs(_cut_moment(X, w, x -> x[i])) < 1e-12
            @test _cut_moment(X, w, x -> x[i]^2) ≈ 1 atol = 1e-6
            @test _cut_moment(X, w, x -> x[i]^4) ≈ 3 atol = 1e-6
            @test _cut_moment(X, w, x -> x[i]^6) ≈ 15 atol = 1e-5
        end
        if n >= 2
            @test abs(_cut_moment(X, w, x -> x[1] * x[2])) < 1e-12
            @test _cut_moment(X, w, x -> x[1]^2 * x[2]^2) ≈ 1 atol = 1e-6
            @test _cut_moment(X, w, x -> x[1]^4 * x[2]^2) ≈ 3 atol = 1e-5
        end
        if n >= 3
            @test _cut_moment(X, w, x -> x[1]^2 * x[2]^2 * x[3]^2) ≈ 1 atol = 1e-6
        end
    end

    X6, = cut_points(6; order = 6)
    counts = _nnz_counts(X6)
    @test counts[0] == 1
    @test counts[1] == 12
    @test counts[2] == 60
    @test counts[6] == 64
    @test X6[:, 2] ≈ [1.9488352799, 0, 0, 0, 0, 0]
    @test X6[:, 3] ≈ [-1.9488352799, 0, 0, 0, 0, 0]

    X7, = cut_points(7; order = 6)
    counts7 = _nnz_counts(X7)
    @test counts7[0] == 1
    @test counts7[1] == 14
    @test counts7[3] == 280
    @test counts7[7] == 128
    @test !haskey(counts7, 2)

    rng = MersenneTwister(11)
    for (order, atol) in ((4, 1e-10), (6, 1e-6))
        n = 4
        x = randn(rng, n)
        A = randn(rng, n, n)
        P = A * A' + 0.3 * Matrix{Float64}(I, n, n)
        ξ, w = cut_points(n; order = order)
        L = cholesky(Hermitian(P)).L
        X = L * ξ .+ reshape(x, :, 1)
        μ = zeros(n)
        Prec = zeros(n, n)
        for i in axes(X, 2)
            μ .+= w[i] .* X[:, i]
        end
        for i in axes(X, 2)
            d = X[:, i] .- μ
            Prec .+= w[i] .* (d * d')
        end
        @test μ ≈ x atol = atol rtol = atol
        @test Symmetric(Prec) ≈ Symmetric(P) atol = atol rtol = atol
    end
end

@testset "unsupported order and dimension" begin
    for (n, order) in ((3, 8), (3, 5), (1, 6), (10, 6), (0, 4))
        err = try
            cut_points(n; order = order)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        if order == 8 || order ∉ (4, 6)
            @test occursin("CUT8", err.msg)
        end
        if order == 6
            @test occursin("2", err.msg)
            @test occursin("9", err.msg)
        end
    end
    @test_throws DimensionMismatch ConjugateUnscentedKalmanFilter(0.0, [0.0, 0.0], [1.0;;])
    @test_throws ArgumentError ConjugateUnscentedKalmanFilter(
        0.0,
        zeros(1),
        fill(1.0, 1, 1);
        order = 6,
    )
end

@testset "constructor copies state" begin
    x = [1.0, -2.0]
    P = [2.0 0.3; 0.3 1.5]
    filt = ConjugateUnscentedKalmanFilter(1.5, x, P)
    x[1] = 0.0
    P[1, 1] = 9.0
    @test filt.t == 1.5
    @test filt.x == [1.0, -2.0]
    @test filt.P[1, 1] == 2.0
    @test filt.nx == 2
    @test filt.order == 4
    @test_throws ArgumentError predict!(filt, (t, z, dt) -> z, Matrix{Float64}(I, 2, 2), NaN)
    @test filt.t == 1.5
    @test filt.x == [1.0, -2.0]
    @test_throws DimensionMismatch predict!(
        filt,
        (t, z, dt) -> [z[1]],
        Matrix{Float64}(I, 2, 2),
        0.1,
    )
    @test filt.t == 1.5
    bad = ConjugateUnscentedKalmanFilter(0.0, [0.0], [-1.0;;])
    @test_throws ArgumentError predict!(bad, (t, z, dt) -> z, [1.0;;], 0.1)
    @test bad.t == 0.0
    @test bad.x == [0.0]
end

@testset "linear Gaussian matches the Kalman update" begin
    F = [1.0 0.1; 0.0 1.0]
    x0 = [1.0, -0.5]
    P0 = [1.0 0.2; 0.2 0.8]
    Q = 0.01 * Matrix{Float64}(I, 2, 2)
    x_pred = F * x0
    P_pred = Symmetric(F * P0 * F' + Q)
    H = [1.0 -0.2]
    R = fill(0.05, 1, 1)
    y = 0.7
    S = Symmetric(H * P_pred * H' + R)
    K = P_pred * H' / S
    x_upd = x_pred + vec(K * (y .- H * x_pred))
    P_upd = Symmetric(P_pred - K * S * K')
    for (order, atol) in ((4, 1e-10), (6, 1e-6))
        filt = ConjugateUnscentedKalmanFilter(0.0, x0, P0; order = order)
        predict!(filt, (t, x, dt) -> F * x, Q, 0.2)
        @test filt.t ≈ 0.2
        @test filt.x ≈ x_pred atol = atol
        @test Symmetric(filt.P) ≈ P_pred atol = atol
        update!(filt, (t, x) -> H * x, y, R)
        @test filt.x ≈ x_upd atol = atol
        @test Symmetric(filt.P) ≈ P_upd atol = atol
        @test issymmetric(filt.P)
        @test all(>(0), eigvals(Symmetric(filt.P)))
    end

    filt1 = ConjugateUnscentedKalmanFilter(0.0, [2.0], fill(0.4, 1, 1); order = 4)
    predict!(filt1, (t, x, dt) -> 0.5 .* x .+ 1.0, fill(0.1, 1, 1), 1.0)
    @test filt1.x ≈ [2.0] atol = 1e-12
    @test filt1.P ≈ fill(0.2, 1, 1) atol = 1e-12
end

@testset "two-body tracking" begin
    μ = 3.986004418e5
    R_E = 6378.137
    ω_E = 7.292115146706979e-5
    a = 7000.0
    inc = deg2rad(51.6)
    lat = deg2rad(28.5)
    lon0 = 0.0

    period = 2π * sqrt(a^3 / μ)
    Δt = period / 10
    n_meas = 6

    function twobody!(du, u, p, t)
        μp = p
        r2 = u[1] * u[1] + u[2] * u[2] + u[3] * u[3]
        μ_inv_r3 = -μp / (r2 * sqrt(r2))
        du[1] = u[4]
        du[2] = u[5]
        du[3] = u[6]
        du[4] = μ_inv_r3 * u[1]
        du[5] = μ_inv_r3 * u[2]
        du[6] = μ_inv_r3 * u[3]
        return nothing
    end

    speed = sqrt(μ / a)
    x0 = [a, 0.0, 0.0, 0.0, speed * cos(inc), speed * sin(inc)]
    truth = solve(
        ODEProblem(twobody!, x0, (0.0, n_meas * Δt), μ),
        Tsit5();
        reltol = 1e-12,
        abstol = 1e-12,
        saveat = Δt,
    )

    function station_state(t)
        cφ = cos(lat)
        sφ = sin(lat)
        λ = lon0 + ω_E * t
        cλ = cos(λ)
        sλ = sin(λ)
        rs = R_E .* (cφ * cλ, cφ * sλ, sφ)
        vs = (ω_E * R_E) .* (-cφ * sλ, cφ * cλ, 0.0)
        return rs, vs
    end

    function range_rangerate(t, x)
        rs, vs = station_state(t)
        ρx = x[1] - rs[1]
        ρy = x[2] - rs[2]
        ρz = x[3] - rs[3]
        ρ = sqrt(ρx * ρx + ρy * ρy + ρz * ρz)
        ρdot = (ρx * (x[4] - vs[1]) + ρy * (x[5] - vs[2]) + ρz * (x[6] - vs[3])) / ρ
        return [ρ, ρdot]
    end

    σ_ρ = 0.01
    σ_ρdot = 1.0e-5
    R = Diagonal([σ_ρ^2, σ_ρdot^2])
    Random.seed!(1)
    ys = map(Δt .* (1:n_meas)) do t
        return range_rangerate(t, truth(t)) .+ [σ_ρ, σ_ρdot] .* randn(2)
    end

    σ_r = 1.0
    σ_v = 1.0e-3
    x_hat0 = x0 .+ [1.0, -0.5, 0.8, 1.0e-3, -0.5e-3, 0.7e-3]
    P0 = Matrix(Diagonal([fill(σ_r^2, 3); fill(σ_v^2, 3)]))
    Q = get_noise(UnbiasedRandomProcess(1.0e-6), Δt)

    function f(t, x, dt)
        sol = solve(
            ODEProblem(twobody!, x, (t, t + dt), μ),
            Tsit5();
            reltol = 1e-12,
            abstol = 1e-12,
            save_everystep = false,
        )
        return sol.u[end]
    end
    h(t, x) = range_rangerate(t, x)

    function run!(filt, nsteps)
        pos = Float64[]
        vel = Float64[]
        for k in 1:nsteps
            predict!(filt, f, Q, Δt)
            update!(filt, h, ys[k], R)
            @test issymmetric(filt.P)
            @test all(>(0), eigvals(Symmetric(filt.P)))
            x_true = truth(filt.t)
            push!(pos, norm(filt.x[1:3] .- x_true[1:3]))
            push!(vel, norm(filt.x[4:6] .- x_true[4:6]))
        end
        return pos, vel
    end

    cut = ConjugateUnscentedKalmanFilter(0.0, x_hat0, P0; order = 4)
    ukf = UnscentedKalmanFilter(0.0, x_hat0, P0)
    pos_cut, vel_cut = run!(cut, n_meas)
    pos_ukf, vel_ukf = run!(ukf, n_meas)
    @test cut.t ≈ n_meas * Δt
    @test pos_cut[end] < 0.5
    @test vel_cut[end] < 1.0e-3
    @test pos_ukf[end] < 0.5
    @test vel_ukf[end] < 1.0e-3
    @test max(pos_cut[end], 1.0e-6) / max(pos_ukf[end], 1.0e-6) < 10
    @test max(pos_ukf[end], 1.0e-6) / max(pos_cut[end], 1.0e-6) < 10
    @test max(vel_cut[end], 1.0e-8) / max(vel_ukf[end], 1.0e-8) < 10
    @test max(vel_ukf[end], 1.0e-8) / max(vel_cut[end], 1.0e-8) < 10

    cut6 = ConjugateUnscentedKalmanFilter(0.0, x_hat0, P0; order = 6)
    pos6, vel6 = run!(cut6, 2)
    @test pos6[end] < 0.5
    @test vel6[end] < 1.0e-3
    @test max(pos6[end], 1.0e-6) / max(pos_ukf[2], 1.0e-6) < 10
    @test max(pos_ukf[2], 1.0e-6) / max(pos6[end], 1.0e-6) < 10
end

end
