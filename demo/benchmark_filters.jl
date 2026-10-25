"""
Elapsed time, process CPU time, and allocations for the demos.

Each measured call builds a new filter and runs the full predict/update
recursion. A warmup call runs first so compilation is not included.
Two-body timings use one warmed run. Linear-kernel timings use the fastest
of several warmed runs. Allocations are those of that same run.
"""

using LinearAlgebra
using Printf
using Random

import SequentialFilters: predict!, update!, covariance

# POSIX clock() reports process CPU time in CLOCKS_PER_SEC units.
const _CLOCKS_PER_SEC = 1_000_000

function process_cpu_seconds()
    ticks = @ccall(clock()::Clong)
    return Float64(ticks) / _CLOCKS_PER_SEC
end

struct RunCost
    elapsed::Float64
    cpu::Float64
    bytes::Int
    allocations::Int
end

function _allocation_count(gcstats)
    return gcstats.poolalloc + gcstats.malloc + gcstats.realloc + gcstats.bigalloc
end

function measure_cost(run!; samples::Int = 1)
    run!()
    best = RunCost(Inf, Inf, 0, 0)
    for _ in 1:samples
        GC.gc()
        cpu0 = process_cpu_seconds()
        stats = @timed run!()
        cpu1 = process_cpu_seconds()
        cost = RunCost(stats.time, cpu1 - cpu0, Int(stats.bytes), _allocation_count(stats.gcstats))
        if cost.elapsed < best.elapsed
            best = cost
        end
    end
    return best
end

function _format_bytes(bytes::Integer)
    value = Float64(bytes)
    if value < 1024
        return @sprintf("%d bytes", bytes)
    elseif value < 1024^2
        return @sprintf("%.3f KiB", value / 1024)
    elseif value < 1024^3
        return @sprintf("%.3f MiB", value / 1024^2)
    else
        return @sprintf("%.3f GiB", value / 1024^3)
    end
end

function _print_cost(name, cost::RunCost)
    @printf(
        "  %-22s elapsed %8.4f s    cpu %8.4f s    %d allocations: %s\n",
        name,
        cost.elapsed,
        cost.cpu,
        cost.allocations,
        _format_bytes(cost.bytes),
    )
    return nothing
end

function _ratio_text(numerator, denominator)
    if !(denominator > 0)
        return "   n/a"
    end
    return @sprintf("%5.2f×", numerator / denominator)
end

function _print_ratio(left::RunCost, right::RunCost)
    @printf(
        "  square-root / covariance    elapsed %s    cpu %s    bytes %s\n",
        _ratio_text(right.elapsed, left.elapsed),
        _ratio_text(right.cpu, left.cpu),
        _ratio_text(right.bytes, left.bytes),
    )
    return nothing
end

function run_ekf(case)
    ekf = ExtendedKalmanFilter(0.0, case.x_hat0, copy(case.P0))
    for k in 1:case.n_meas
        predict!(ekf, case.dynamics, case.process_noise, case.Δt)
        update!(ekf, case.measurements, case.ys[k], case.R)
    end
    return ekf
end

function run_srekf(case)
    filt = SquareRootExtendedKalmanFilter(0.0, case.x_hat0, case.P0)
    for k in 1:case.n_meas
        predict!(filt, case.dynamics, case.process_noise, case.Δt)
        update!(filt, case.h, case.H, case.ys[k], case.R)
    end
    return filt
end

function run_ukf(case)
    filt = UnscentedKalmanFilter(0.0, case.x_hat0, case.P0)
    for k in 1:case.n_meas
        predict!(filt, case.dynamics, case.process_noise, case.Δt)
        update!(filt, case.h, case.ys[k], case.R)
    end
    return filt
end

function run_srukf(case)
    filt = SquareRootUnscentedKalmanFilter(0.0, case.x_hat0, case.P0)
    for k in 1:case.n_meas
        predict!(filt, case.dynamics, case.process_noise, case.Δt)
        update!(filt, case.h, case.ys[k], case.R)
    end
    return filt
end

function run_srckf(case)
    filt = SquareRootCubatureKalmanFilter(0.0, case.x_hat0, case.P0)
    for k in 1:case.n_meas
        predict!(filt, case.dynamics, case.process_noise, case.Δt)
        update!(filt, case.h, case.ys[k], case.R)
    end
    return filt
end

function _linear_model()
    Δt = 0.5
    F = [1.0 Δt 0.0 0.0 0.0 0.0;
         0.0 1.0 0.0 0.0 0.0 0.0;
         0.0 0.0 1.0 Δt 0.0 0.0;
         0.0 0.0 0.0 1.0 0.0 0.0;
         0.0 0.0 0.0 0.0 1.0 Δt;
         0.0 0.0 0.0 0.0 0.0 1.0]
    H = zeros(2, 6)
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    Q = Matrix(1.0e-4 * I(6))
    R = Matrix(1.0e-2 * I(2))
    x0 = [1.0, 0.1, -0.5, 0.05, 0.2, -0.02]
    P0 = Matrix{Float64}(I, 6, 6)
    y = [1.1, 0.08]
    return (; F, H, Q, R, x0, P0, y, nsteps = 5000)
end

function run_discrete_covariance_ekf()
    model = _linear_model()
    x = copy(model.x0)
    P = copy(model.P0)
    for _ in 1:model.nsteps
        x = model.F * x
        mapped = model.F * P * model.F' + model.Q
        P = (mapped + mapped') ./ 2
        yhat = model.H * x
        S = model.H * P * model.H' + model.R
        K = (P * model.H') / S
        x = x + K * (model.y - yhat)
        IKH = I(6) - K * model.H
        updated = IKH * P * IKH' + K * model.R * K'
        P = (updated + updated') ./ 2
    end
    return x, P
end

function run_discrete_srekf()
    model = _linear_model()
    filt = SquareRootExtendedKalmanFilter(0.0, model.x0, model.P0)
    f = (t, x, dt) -> model.F * x
    h = (t, x) -> model.H * x
    dynamics = DiscreteDynamicsModel(f; n = 6, jacobian = (t, x, dt) -> model.F)
    noise = AdditiveProcessNoise(model.Q)
    for _ in 1:model.nsteps
        predict!(filt, dynamics, noise, 1.0)
        update!(filt, h, model.H, model.y, model.R)
    end
    return filt
end

function _linear_step(model)
    dynamics = DiscreteDynamicsModel((t, x, dt) -> model.F * x; n = 6)
    noise = AdditiveProcessNoise(model.Q)
    h = (t, x) -> model.H * x
    return dynamics, noise, h
end

function run_discrete_ukf()
    model = _linear_model()
    filt = UnscentedKalmanFilter(0.0, model.x0, model.P0)
    dynamics, noise, h = _linear_step(model)
    for _ in 1:model.nsteps
        predict!(filt, dynamics, noise, 1.0)
        update!(filt, h, model.y, model.R)
    end
    return filt
end

function run_cut(case)
    filt = ConjugateUnscentedKalmanFilter(0.0, case.x_hat0, case.P0; order = 4)
    for k in 1:case.n_meas
        predict!(filt, case.dynamics, case.process_noise, case.Δt)
        update!(filt, case.h, case.ys[k], case.R)
    end
    return filt
end

function run_srcut(case)
    filt = SquareRootConjugateUnscentedKalmanFilter(0.0, case.x_hat0, case.P0; order = 4)
    for k in 1:case.n_meas
        predict!(filt, case.dynamics, case.process_noise, case.Δt)
        update!(filt, case.h, case.ys[k], case.R)
    end
    return filt
end

function run_discrete_cut()
    model = _linear_model()
    filt = ConjugateUnscentedKalmanFilter(0.0, model.x0, model.P0; order = 4)
    dynamics, noise, h = _linear_step(model)
    for _ in 1:model.nsteps
        predict!(filt, dynamics, noise, 1.0)
        update!(filt, h, model.y, model.R)
    end
    return filt
end

function run_discrete_srcut()
    model = _linear_model()
    filt = SquareRootConjugateUnscentedKalmanFilter(0.0, model.x0, model.P0; order = 4)
    dynamics, noise, h = _linear_step(model)
    for _ in 1:model.nsteps
        predict!(filt, dynamics, noise, 1.0)
        update!(filt, h, model.y, model.R)
    end
    return filt
end

function run_discrete_srukf()
    model = _linear_model()
    filt = SquareRootUnscentedKalmanFilter(0.0, model.x0, model.P0)
    dynamics, noise, h = _linear_step(model)
    for _ in 1:model.nsteps
        predict!(filt, dynamics, noise, 1.0)
        update!(filt, h, model.y, model.R)
    end
    return filt
end

function run_discrete_srckf()
    model = _linear_model()
    filt = SquareRootCubatureKalmanFilter(0.0, model.x0, model.P0)
    dynamics, noise, h = _linear_step(model)
    for _ in 1:model.nsteps
        predict!(filt, dynamics, noise, 1.0)
        update!(filt, h, model.y, model.R)
    end
    return filt
end

function run_particle(case; n_particles::Integer = 64)
    filt = ParticleFilter(0.0, case.x_hat0, case.P0; n_particles = n_particles, rng = MersenneTwister(1))
    for k in 1:case.n_meas
        predict!(filt, case.dynamics, case.process_noise, case.Δt)
        update!(filt, case.measurements, case.ys[k], case.R)
    end
    return filt
end

function run_discrete_particle()
    model = _linear_model()
    filt = ParticleFilter(0.0, model.x0, model.P0; n_particles = 32, rng = MersenneTwister(1))
    dynamics, noise, h = _linear_step(model)
    for _ in 1:200
        predict!(filt, dynamics, noise, 1.0)
        update!(filt, h, model.y, model.R)
    end
    return filt
end

"""
    print_filter_costs(case, pair)

`pair` is `:ekf`, `:ukf`, `:cut`, `:srckf`, or `:particle`. The first three print
two-body and linear-kernel costs for the covariance-form filter and its
square-root form. The CUT pair uses order 4. `:srckf` times the square-root
cubature filter alone. `:particle` times a 64-particle two-body run and a
32-particle linear kernel of 200 steps.
"""
function print_filter_costs(case, pair::Symbol)
    println()
    println("Performance: warmup excluded. CPU time is process CPU from clock().")
    println("Process CPU time sums every thread, so it can exceed elapsed time.")
    println("Allocations follow the Julia @time count (pool + malloc + realloc + big).")
    if pair == :ekf
        println("two-body recursion (same propagation as the EKF demo, one warmed run)")
        covariance_cost = measure_cost(() -> run_ekf(case); samples = 1)
        square_root_cost = measure_cost(() -> run_srekf(case); samples = 1)
        _print_cost("EKF", covariance_cost)
        _print_cost("square-root EKF", square_root_cost)
        _print_ratio(covariance_cost, square_root_cost)
        println("linear kernel (6 states, 2 measurements, 5000 steps, fastest of 5 runs)")
        covariance_cost = measure_cost(run_discrete_covariance_ekf; samples = 5)
        square_root_cost = measure_cost(run_discrete_srekf; samples = 5)
        _print_cost("covariance EKF", covariance_cost)
        _print_cost("square-root EKF", square_root_cost)
        _print_ratio(covariance_cost, square_root_cost)
    elseif pair == :ukf
        println("two-body recursion (same sigma-point propagation, one warmed run)")
        covariance_cost = measure_cost(() -> run_ukf(case); samples = 1)
        square_root_cost = measure_cost(() -> run_srukf(case); samples = 1)
        _print_cost("UKF", covariance_cost)
        _print_cost("square-root UKF", square_root_cost)
        _print_ratio(covariance_cost, square_root_cost)
        println("linear kernel (6 states, 2 measurements, 5000 steps, fastest of 5 runs)")
        covariance_cost = measure_cost(run_discrete_ukf; samples = 5)
        square_root_cost = measure_cost(run_discrete_srukf; samples = 5)
        _print_cost("UKF", covariance_cost)
        _print_cost("square-root UKF", square_root_cost)
        _print_ratio(covariance_cost, square_root_cost)
    elseif pair == :cut
        println("two-body recursion (same CUT4 sigma points, one warmed run)")
        covariance_cost = measure_cost(() -> run_cut(case); samples = 1)
        square_root_cost = measure_cost(() -> run_srcut(case); samples = 1)
        _print_cost("CUT", covariance_cost)
        _print_cost("square-root CUT", square_root_cost)
        _print_ratio(covariance_cost, square_root_cost)
        println("linear kernel (CUT4, 6 states, 2 measurements, 5000 steps, fastest of 5 runs)")
        covariance_cost = measure_cost(run_discrete_cut; samples = 5)
        square_root_cost = measure_cost(run_discrete_srcut; samples = 5)
        _print_cost("CUT", covariance_cost)
        _print_cost("square-root CUT", square_root_cost)
        _print_ratio(covariance_cost, square_root_cost)
    elseif pair == :srckf
        println("two-body recursion (2n cubature points, one warmed run)")
        square_root_cost = measure_cost(() -> run_srckf(case); samples = 1)
        _print_cost("square-root CKF", square_root_cost)
        println("linear kernel (6 states, 2 measurements, 5000 steps, fastest of 5 runs)")
        square_root_cost = measure_cost(run_discrete_srckf; samples = 5)
        _print_cost("square-root CKF", square_root_cost)
        println("There is no covariance-form cubature filter in this package, so no ratio is printed.")
    elseif pair == :particle
        println("two-body recursion (64 particles, serial ensemble, one warmed run)")
        particle_cost = measure_cost(() -> run_particle(case); samples = 1)
        _print_cost("particle filter", particle_cost)
        println("linear kernel (32 particles, 6 states, 200 steps, fastest of 3 runs)")
        particle_cost = measure_cost(run_discrete_particle; samples = 3)
        _print_cost("particle filter", particle_cost)
    else
        throw(ArgumentError("pair must be :ekf, :ukf, :cut, :srckf, or :particle, got $pair"))
    end
    println("On the two-body run the measured filters share the ODE propagation.")
    println("The linear kernel is the predict/update cost without that propagation.")
    return nothing
end
