"""
Monte Carlo comparison of every SequentialFilters filter on the two-body scenario.

    julia --project=experiments/square-root-benchmark experiments/square-root-benchmark/run_twobody.jl

Optional arguments: `--M`, `--sweep-M`, `--seed`, and `--skip-sweep`.
The measurement-noise sweep uses seed + 1. A non-positive `--sweep-M` skips it.
"""

import Pkg
Pkg.activate(@__DIR__)

ENV["GKSwstype"] = "nul"

using CairoMakie
using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq
using Printf
using Random

import SequentialFilters

include(joinpath(@__DIR__, "scenario.jl"))
include(joinpath(@__DIR__, "scenarios", "twobody.jl"))
include(joinpath(@__DIR__, "filters.jl"))
include(joinpath(@__DIR__, "pcrlb.jl"))
include(joinpath(@__DIR__, "metrics.jl"))
include(joinpath(@__DIR__, "runner.jl"))
include(joinpath(@__DIR__, "report.jl"))

const OUTPUT_DIR = joinpath(@__DIR__, "output")
const NOISE_SCALES = (0.5, 1.0, 2.0, 5.0)

function parse_cli(args)
    M = 100
    sweep_M = 20
    seed = 20261024
    skip_sweep = false
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--M"
            M = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--sweep-M"
            sweep_M = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--seed"
            seed = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--skip-sweep"
            skip_sweep = true
            i += 1
        else
            error("unrecognized argument $arg")
        end
    end
    return (; M, sweep_M, seed, skip_sweep)
end

function posterior_bounds(scenario, information)
    position = crlb_rmse(information, scenario.blocks.position)
    velocity = crlb_rmse(information, scenario.blocks.velocity)
    posterior = 2:scenario.n_steps+1
    return (;
        position,
        velocity,
        position_average = SequentialFilters.time_averaged_rmse(position[posterior]),
        velocity_average = SequentialFilters.time_averaged_rmse(velocity[posterior]),
    )
end

function run_batch(scenario, specs; M, seed, warmup = true)
    information = posterior_information(scenario)
    bounds = posterior_bounds(scenario, information)
    results = monte_carlo(scenario, specs; M = M, seed = seed, warmup = warmup)
    rows = summarize(scenario, results)
    return (; information, bounds, results, rows)
end

function run_sensitivity(specs, output_dir; M, seed)
    records = NamedTuple[]
    println("measurement-noise sweep  M = $M  seed = $seed  scales = $(collect(NOISE_SCALES))")
    flush(stdout)
    for (index, scale) in enumerate(NOISE_SCALES)
        scenario = twobody_scenario(; measurement_noise_scale = scale)
        batch = run_batch(scenario, specs; M = M, seed = seed, warmup = index == 1)
        println("scale $scale")
        println(format_summary(batch.rows))
        println(
            @sprintf(
                "PCRLB position %.6f km    velocity %.6e km/s",
                batch.bounds.position_average,
                batch.bounds.velocity_average,
            ),
        )
        flush(stdout)
        for row in batch.rows
            push!(
                records,
                (;
                    scale = Float64(scale),
                    name = row.name,
                    position_rmse = row.position_rmse,
                    velocity_rmse = row.velocity_rmse,
                    divergence_rate = row.divergence_rate,
                    runtime = row.runtime,
                    pcrlb_position = batch.bounds.position_average,
                    pcrlb_velocity = batch.bounds.velocity_average,
                ),
            )
        end
    end
    path = joinpath(output_dir, "sensitivity_measurement_noise.csv")
    write_sensitivity_csv(path, records)
    println("wrote $path")
    return records
end

function main()
    cli = parse_cli(ARGS)
    mkpath(OUTPUT_DIR)
    specs = package_filters()
    scenario = twobody_scenario()
    println("filters: ", join((spec.name for spec in specs), ", "))
    println("seed = $(cli.seed)    M = $(cli.M)    steps = $(scenario.n_steps)    Δt = $(scenario.Δt) s")
    println("divergence threshold = $(scenario.divergence_threshold) km on the position error")
    println("initial-error scale, sample interval, orbit count, and model mismatch are scenario arguments")
    flush(stdout)
    batch = run_batch(scenario, specs; M = cli.M, seed = cli.seed, warmup = true)
    header = """
    Two-body Monte Carlo
    M = $(cli.M)
    seed = $(cli.seed)
    divergence threshold = $(scenario.divergence_threshold) km (position)
    time average = posterior measurement instants
    PCRLB position = $(batch.bounds.position_average) km
    PCRLB velocity = $(batch.bounds.velocity_average) km/s
    """
    text = format_summary(batch.rows)
    println(header)
    println(text)
    write(joinpath(OUTPUT_DIR, "twobody_table.txt"), header * "\n" * text * "\n")
    write_summary_csv(joinpath(OUTPUT_DIR, "twobody_table.csv"), batch.rows)
    save_rmse_figure(
        joinpath(OUTPUT_DIR, "experiments_rmse.png"),
        scenario,
        batch.results,
        batch.bounds.position,
        batch.bounds.velocity,
    )
    save_nees_figure(joinpath(OUTPUT_DIR, "experiments_nees.png"), scenario, batch.results, cli.M)
    save_nis_figure(joinpath(OUTPUT_DIR, "experiments_nis.png"), scenario, batch.results, cli.M)
    save_autocorrelation_figure(
        joinpath(OUTPUT_DIR, "experiments_autocorrelation.png"),
        scenario,
        batch.results,
    )
    println("wrote $(OUTPUT_DIR)")
    if !cli.skip_sweep && cli.sweep_M > 0
        run_sensitivity(specs, OUTPUT_DIR; M = cli.sweep_M, seed = cli.seed + 1)
    end
    return nothing
end

main()
