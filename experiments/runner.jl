"""
Monte Carlo runner.

One trial draws a truth and a measurement sequence and feeds that same draw to
every filter. `M` and the seed are arguments. Compilation is excluded by one
untimed warmup per filter. A non-finite estimate, covariance, or thrown step
stops that run and marks it diverged. A finite error past the scenario
threshold also marks the run, and those finite samples stay in the RMSE, NEES,
and NIS records.
"""

struct TrialResult
    name::String
    errors::Array{Float64,3}
    nees::Matrix{Float64}
    nis::Matrix{Float64}
    innovations::Array{Float64,3}
    diverged::Vector{Bool}
    runtime::Float64
end

function _record_posterior!(errors, nees_hist, scenario, truth, xhat, P, k)
    finite = all(isfinite, xhat) && all(isfinite, P)
    if !finite
        return true
    end
    residual = SequentialFilters.estimation_error(truth, xhat)
    errors[:, k] = residual
    nees_hist[k] = SequentialFilters.nees(residual, P)
    position = view(residual, scenario.divergence_indices)
    return SequentialFilters.is_diverged(
        position;
        threshold = scenario.divergence_threshold,
        estimate = xhat,
        covariance = P,
    )
end

function run_filter(scenario::Scenario, spec::FilterSpec, draw)
    nx = scenario.nx
    ny = scenario.ny
    n_steps = scenario.n_steps
    errors = fill(NaN, nx, n_steps + 1)
    nees_hist = fill(NaN, n_steps + 1)
    nis_hist = fill(NaN, n_steps)
    innovations = fill(NaN, ny, n_steps)
    filt = spec.build(0.0, draw.xhat0, draw.P0)
    xhat = spec.mean(filt)
    P = spec.covariance(filt)
    diverged = _record_posterior!(errors, nees_hist, scenario, draw.truth[:, 1], xhat, P, 1)
    stopped = diverged && !(all(isfinite, xhat) && all(isfinite, P))
    if !stopped
        for k in 1:n_steps
            y = draw.ys[k]
            try
                spec.predict(filt, scenario)
                innovation, S = spec.innovation(filt, scenario, y)
                innovations[:, k] = innovation
                if all(isfinite, innovation) && all(isfinite, S)
                    nis_hist[k] = SequentialFilters.nis(innovation, S)
                else
                    diverged = true
                end
                spec.update(filt, scenario, y)
                xhat = spec.mean(filt)
                P = spec.covariance(filt)
                step_diverged = _record_posterior!(
                    errors, nees_hist, scenario, draw.truth[:, k + 1], xhat, P, k + 1,
                )
                diverged |= step_diverged
                if !(all(isfinite, xhat) && all(isfinite, P))
                    stopped = true
                    break
                end
            catch err
                if err isa InterruptException
                    rethrow()
                end
                diverged = true
                stopped = true
                println("$(spec.name) run stopped: $(sprint(showerror, err))")
                break
            end
        end
    end
    return errors, nees_hist, nis_hist, innovations, diverged
end

function monte_carlo(scenario::Scenario, specs; M::Integer, seed::Integer, warmup::Bool = true)
    m = Int(M)
    m >= 1 || throw(ArgumentError("M must be positive, got $M"))
    rng = MersenneTwister(Int(seed))
    draws = [scenario.draw(rng) for _ in 1:m]
    if warmup
        warm_draw = scenario.draw(MersenneTwister(Int(seed) + 1_000_003))
        for spec in specs
            println("warmup $(spec.name)")
            flush(stdout)
            run_filter(scenario, spec, warm_draw)
        end
    end

    results = TrialResult[]
    for spec in specs
        nx = scenario.nx
        ny = scenario.ny
        n_steps = scenario.n_steps
        errors = Array{Float64,3}(undef, nx, m, n_steps + 1)
        nees = Matrix{Float64}(undef, m, n_steps + 1)
        nis = Matrix{Float64}(undef, m, n_steps)
        innovations = Array{Float64,3}(undef, ny, m, n_steps)
        diverged = falses(m)
        elapsed = 0.0
        for i in 1:m
            if i == 1 || i == m || i % 10 == 0
                println("$(spec.name)  run $i / $m")
                flush(stdout)
            end
            started = time_ns()
            err_i, nees_i, nis_i, innov_i, flag = run_filter(scenario, spec, draws[i])
            elapsed += (time_ns() - started) / 1e9
            errors[:, i, :] = err_i
            nees[i, :] = nees_i
            nis[i, :] = nis_i
            innovations[:, i, :] = innov_i
            diverged[i] = flag
        end
        push!(results, TrialResult(spec.name, errors, nees, nis, innovations, diverged, elapsed / m))
    end
    return results
end
