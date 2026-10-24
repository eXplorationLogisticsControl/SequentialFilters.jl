"""
Coordinated-turn tracking with a scaled unscented Kalman filter.

The target state is `[x, y, vx, vy, ω]`. Motion is a discrete coordinated
turn. A sensor at the origin measures range and bearing. Process and
measurement noise are additive; the filter does not use Jacobians.

Run from the repository root:

    julia --project=. demo/demo_ukf.jl
"""

using LinearAlgebra
using Printf
using Random

using SequentialFilters

"""Discrete coordinated turn. State is `[x, y, vx, vy, ω]`."""
function coordinated_turn(_t, state, Δt)
    px, py, vx, vy, ω = state
    θ = ω * Δt
    if abs(θ) < 1e-9
        return [px + vx * Δt, py + vy * Δt, vx, vy, ω]
    end
    s, c = sincos(θ)
    s_over_ω = s / ω
    one_c_over_ω = (1 - c) / ω
    return [
        px + s_over_ω * vx - one_c_over_ω * vy,
        py + one_c_over_ω * vx + s_over_ω * vy,
        c * vx - s * vy,
        s * vx + c * vy,
        ω,
    ]
end

function range_bearing(_t, state)
    return [hypot(state[1], state[2]), atan(state[2], state[1])]
end

function run_demo()
    Δt = 0.5
    nsteps = 40
    x_true = [200.0, 50.0, 8.0, 0.0, 0.08]
    x_hat0 = [185.0, 60.0, 7.0, 1.5, 0.06]
    P0 = Diagonal([20.0, 20.0, 2.0, 2.0, 0.05] .^ 2)
    Q = Diagonal([0.15, 0.15, 0.08, 0.08, 0.002] .^ 2)
    R = Diagonal([2.0, deg2rad(1.0)] .^ 2)

    rng = MersenneTwister(7)
    Qchol = cholesky(Hermitian(Matrix(Q))).L
    Rchol = cholesky(Hermitian(Matrix(R))).L
    truth = [copy(x_true)]
    ys = Vector{Float64}[]
    t = 0.0
    x = copy(x_true)
    for _ in 1:nsteps
        x = coordinated_turn(t, x, Δt) .+ Qchol * randn(rng, 5)
        t += Δt
        y = range_bearing(t, x) .+ Rchol * randn(rng, 2)
        push!(truth, copy(x))
        push!(ys, y)
    end

    ukf = UnscentedKalmanFilter(0.0, x_hat0, P0)
    open_loop = copy(x_hat0)
    t_ol = 0.0
    pos_err = Float64[]
    vel_err = Float64[]
    rate_err = Float64[]
    for k in 1:nsteps
        predict!(ukf, coordinated_turn, Q, Δt)
        update!(ukf, range_bearing, ys[k], R)
        open_loop = coordinated_turn(t_ol, open_loop, Δt)
        t_ol += Δt
        push!(pos_err, norm(ukf.x[1:2] .- truth[k + 1][1:2]))
        push!(vel_err, norm(ukf.x[3:4] .- truth[k + 1][3:4]))
        push!(rate_err, abs(ukf.x[5] - truth[k + 1][5]))
    end

    rmse(v) = sqrt(sum(abs2, v) / length(v))
    min_eig = minimum(eigvals(Symmetric(ukf.P)))
    open_loop_pos = norm(open_loop[1:2] .- truth[end][1:2])

    println("UKF coordinated-turn demo")
    @printf(
        "parameters: alpha=%.1e, beta=%g, kappa=%g\n",
        ukf.alpha,
        ukf.beta,
        ukf.kappa,
    )
    @printf("state dimension: %d, measurement dimension: %d\n", ukf.nx, length(ys[1]))
    @printf("steps: %d, dt: %g\n", nsteps, Δt)
    @printf("position RMSE: %.4f\n", rmse(pos_err))
    @printf("velocity RMSE: %.4f\n", rmse(vel_err))
    @printf("turn-rate RMSE: %.6f\n", rmse(rate_err))
    @printf("final position error: %.4f\n", pos_err[end])
    @printf("final velocity error: %.4f\n", vel_err[end])
    @printf("final turn-rate error: %.6f\n", rate_err[end])
    @printf("open-loop final position error: %.4f\n", open_loop_pos)
    @printf("covariance symmetric: %s\n", issymmetric(ukf.P) ? "yes" : "no")
    @printf("minimum eigenvalue: %.6e\n", min_eig)
    return nothing
end

run_demo()
