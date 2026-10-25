"""
Elapsed time, process CPU time, and allocations for the Lorenz-63 demos.

Include `benchmark_filters.jl` first. `measure_cost` warms up once, then times
one Lorenz recursion. The printed lines match that helper: elapsed time,
process CPU time, and the Julia allocation count.
"""

function run_lorenz_ekf(case)
    filt = ExtendedKalmanFilter(0.0, case.x_hat0, case.P0)
    for k in 1:case.n_steps
        predict!(filt, case.dynamics, case.process_noise, case.Δt)
        update!(filt, case.measurements, case.ys[k], case.R)
    end
    return filt
end

function run_lorenz_srekf(case)
    filt = SquareRootExtendedKalmanFilter(0.0, case.x_hat0, case.P0)
    for k in 1:case.n_steps
        predict!(filt, case.dynamics, case.process_noise, case.Δt)
        update!(filt, case.measurements, case.ys[k], case.R)
    end
    return filt
end

function run_lorenz_ukf(case)
    filt = UnscentedKalmanFilter(0.0, case.x_hat0, case.P0)
    for k in 1:case.n_steps
        predict!(filt, case.dynamics, case.process_noise, case.Δt)
        update!(filt, case.h, case.ys[k], case.R)
    end
    return filt
end

function run_lorenz_srukf(case)
    filt = SquareRootUnscentedKalmanFilter(0.0, case.x_hat0, case.P0)
    for k in 1:case.n_steps
        predict!(filt, case.dynamics, case.process_noise, case.Δt)
        update!(filt, case.h, case.ys[k], case.R)
    end
    return filt
end

function run_lorenz_cut(case)
    filt = ConjugateUnscentedKalmanFilter(0.0, case.x_hat0, case.P0; order = 4)
    for k in 1:case.n_steps
        predict!(filt, case.dynamics, case.process_noise, case.Δt)
        update!(filt, case.h, case.ys[k], case.R)
    end
    return filt
end

function run_lorenz_srcut(case)
    filt = SquareRootConjugateUnscentedKalmanFilter(0.0, case.x_hat0, case.P0; order = 4)
    for k in 1:case.n_steps
        predict!(filt, case.dynamics, case.process_noise, case.Δt)
        update!(filt, case.h, case.ys[k], case.R)
    end
    return filt
end

function run_lorenz_srckf(case)
    filt = SquareRootCubatureKalmanFilter(0.0, case.x_hat0, case.P0)
    for k in 1:case.n_steps
        predict!(filt, case.dynamics, case.process_noise, case.Δt)
        update!(filt, case.h, case.ys[k], case.R)
    end
    return filt
end

function run_lorenz_particle(case)
    filt = ParticleFilter(
        0.0,
        case.x_hat0,
        case.P0;
        n_particles = case.n_particles,
        rng = MersenneTwister(1),
    )
    for k in 1:case.n_steps
        predict!(filt, case.dynamics, case.process_noise, case.Δt)
        update!(filt, case.measurements, case.ys[k], case.R)
    end
    return filt
end

"""
    print_lorenz_costs(case, pair)

`pair` is `:ekf`, `:ukf`, `:cut`, `:srckf`, or `:particle`. The first three
time the covariance-form filter and its square-root form on this Lorenz case.
`:srckf` times the square-root cubature filter alone. `:particle` times the
particle filter at `case.n_particles`.
"""
function print_lorenz_costs(case, pair::Symbol)
    println()
    println("Performance: warmup excluded. CPU time is process CPU from clock().")
    println("Process CPU time sums every thread, so it can exceed elapsed time.")
    println("Allocations follow the Julia @time count (pool + malloc + realloc + big).")
    if pair == :ekf
        println("Lorenz-63 recursion (same propagation as the EKF demo, one warmed run)")
        covariance_cost = measure_cost(() -> run_lorenz_ekf(case); samples = 1)
        square_root_cost = measure_cost(() -> run_lorenz_srekf(case); samples = 1)
        _print_cost("EKF", covariance_cost)
        _print_cost("square-root EKF", square_root_cost)
        _print_ratio(covariance_cost, square_root_cost)
    elseif pair == :ukf
        println("Lorenz-63 recursion (same sigma-point propagation, one warmed run)")
        covariance_cost = measure_cost(() -> run_lorenz_ukf(case); samples = 1)
        square_root_cost = measure_cost(() -> run_lorenz_srukf(case); samples = 1)
        _print_cost("UKF", covariance_cost)
        _print_cost("square-root UKF", square_root_cost)
        _print_ratio(covariance_cost, square_root_cost)
    elseif pair == :cut
        println("Lorenz-63 recursion (same CUT4 points, one warmed run)")
        covariance_cost = measure_cost(() -> run_lorenz_cut(case); samples = 1)
        square_root_cost = measure_cost(() -> run_lorenz_srcut(case); samples = 1)
        _print_cost("CUT", covariance_cost)
        _print_cost("square-root CUT", square_root_cost)
        _print_ratio(covariance_cost, square_root_cost)
    elseif pair == :srckf
        println("Lorenz-63 recursion (2n cubature points, one warmed run)")
        square_root_cost = measure_cost(() -> run_lorenz_srckf(case); samples = 1)
        _print_cost("square-root CKF", square_root_cost)
        println("There is no covariance-form cubature filter in this package, so no ratio is printed.")
    elseif pair == :particle
        println("Lorenz-63 recursion ($(case.n_particles) particles, serial ensemble, one warmed run)")
        particle_cost = measure_cost(() -> run_lorenz_particle(case); samples = 1)
        _print_cost("particle filter", particle_cost)
    else
        throw(ArgumentError("pair must be :ekf, :ukf, :cut, :srckf, or :particle, got $pair"))
    end
    println("The timed run is the predict/update recursion. The figure above also stores the ODESolution between steps.")
    return nothing
end
