"""
Square-root CUT benchmarks: ill-conditioned tracking and long-arc angles-only OD.

    julia --project=experiments experiments/run_srcut_benchmarks.jl

Optional arguments: `--M`, `--seed`, `--illcond-only`, `--longarc-only`.
Default Monte Carlo size is 200. The same draws are reused by every filter in
a setting. Float32 and Float64 share those draws. Output tables and the
working figures go to `experiments/output/srcut/`, which is not committed.
The six main PNGs are copied to the project media directory.
"""

import Pkg
Pkg.activate(@__DIR__)

ENV["GKSwstype"] = "nul"

using Printf
using Random

import SequentialFilters

include(joinpath(@__DIR__, "srcut_lib.jl"))

const OUTPUT_DIR = joinpath(@__DIR__, "output", "srcut")
const LEO_GAPS_H = (2.0, 6.0, 12.0, 24.0, 48.0)
const HEO_GAPS_H = (12.0, 24.0, 48.0, 96.0)

function parse_cli(args)
    M = 200
    seed = 20261025
    illcond = true
    longarc = true
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--M"
            M = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--seed"
            seed = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--illcond-only"
            longarc = false
            i += 1
        elseif arg == "--longarc-only"
            illcond = false
            i += 1
        else
            error("unrecognized argument $arg")
        end
    end
    M >= 1 || error("M must be positive")
    return (; M, seed, illcond, longarc)
end

function check_propagators()
    x = circular_state(400.0, deg2rad(51.6))
    fast = propagate(x, 3600.0, B_LEO)
    generic = propagate_generic(x, 3600.0, B_LEO)
    gap = norm(fast .- generic)
    println("propagator agreement over 1 h: |fast − generic| = $gap km")
    gap < 1e-6 || error("fast and generic propagators disagree")
    Q = discrete_process_covariance(4 * 86400.0)
    factor = cholesky(Hermitian(Q); check = false)
    println("4-day process-noise Cholesky success = $(issuccess(factor))  cond = $(safe_cond(Q))")
    issuccess(factor) || error("process-noise covariance is not a stable Cholesky factor at 4 days")
    return nothing
end

function run_illcond_study(M, seed)
    specs = bench_filters(5; orders = (4, 6))
    rows = Summary[]
    times = collect(0.0:ILL_DT:(ILL_STEPS * ILL_DT))
    println("ill-conditioned filters: ", join(["$(s.name)/$(s.precision)" for s in specs], ", "))
    for (index, ratio) in enumerate(ILL_RATIOS)
        σ_position = ratio * ILL_SIGMA_RANGE
        println("ratio $ratio  σ_position = $σ_position km  M = $M")
        flush(stdout)
        draws = [draw_illcond(MersenneTwister(seed + 1000 * index + i), σ_position) for i in 1:M]
        # One shared stream would also be valid. Independent streams with a
        # documented seed keep a failed draw from shifting every later draw,
        # and every filter still sees `draws` unchanged.
        bounds = []
        for draw in draws
            try
                push!(bounds, ill_crlb(draw))
            catch err
                println("illcond CRLB skipped a draw: $(sprint(showerror, err))")
            end
        end
        covariances = ensemble_crlb(bounds)
        crlb = crlb_position(covariances, 1:2)
        warm = draw_illcond(MersenneTwister(seed + 9_000_000 + index), σ_position)
        meta = (;
            case = "illcond",
            setting = "ratio=$ratio",
            ratio = ratio,
            requested_gap_h = NaN,
            actual_gap_h = NaN,
        )
        for spec in specs
            println("  warmup $(spec.name) $(spec.precision)")
            flush(stdout)
            run_illcond(spec, [warm], σ_position)
            println("  run $(spec.name) $(spec.precision)")
            flush(stdout)
            raw = run_illcond(spec, draws, σ_position)
            summary = summarize_filter(spec, raw, times, 1:2, 5:5, 5, crlb, 0, length(times), meta)
            push!(rows, summary)
            @printf(
                "    pos RMSE %s   NEES %s   factor fail %.3f   diverge %.3f   s/step %.3e\n",
                fmt(summary.rmse), fmt(summary.nees), summary.factor_fail, summary.diverge, summary.runtime,
            )
            flush(stdout)
        end
    end
    return rows
end

function prepare_masks()
    leo = (;
        name = "LEO",
        x0 = circular_state(400.0, deg2rad(51.6)),
        B = B_LEO,
    )
    heo = (; name = "HEO", x0 = heo_state(), B = B_HEO)
    masks = []
    for body in (leo, heo)
        describe_orbit(body.name, body.x0)
        horizon = (maximum(body.name == "LEO" ? LEO_GAPS_H : HEO_GAPS_H) + 24.0 * 8) * 3600
        println("visibility mask $(body.name) horizon = $(horizon / 3600) h")
        flush(stdout)
        times, elevs = visibility_mask(body.x0, body.B, horizon)
        push!(masks, (; body.name, body.x0, body.B, times, elevs))
    end
    return masks
end

function run_longarc_study(M, seed, masks)
    specs = bench_filters(6; precisions = (Float64,), orders = (4,), ukf = true)
    rows = Summary[]
    println("long-arc filters: ", join([s.name for s in specs], ", "))
    plans = []
    for mask in masks
        gaps = mask.name == "LEO" ? LEO_GAPS_H : HEO_GAPS_H
        σ_r = mask.name == "LEO" ? 2.0 : 10.0
        σ_v = mask.name == "LEO" ? 2.0e-3 : 5.0e-3
        for gap_h in gaps
            case = build_orbit_case(mask, gap_h * 3600; σ_r = σ_r, σ_v = σ_v)
            if case === nothing
                println("$(mask.name) gap $(gap_h) h: no second pass inside the visibility horizon")
                continue
            end
            @printf(
                "%s requested %.1f h  actual coast %.2f h  steps %d  epochs %d\n",
                mask.name, gap_h, case.actual_gap / 3600, length(case.steps), length(case.times),
            )
            push!(plans, (; case, gap_h, σ_r, σ_v))
        end
    end
    for (index, plan) in enumerate(plans)
        case = plan.case
        println("long-arc $(case.name) actual gap $(case.actual_gap / 3600) h  M = $M")
        flush(stdout)
        draws = [draw_orbit(MersenneTwister(seed + 50_000 + 1000 * index + i), case) for i in 1:M]
        bounds = []
        for draw in draws
            try
                push!(bounds, orbit_crlb(case, draw))
            catch err
                println("long-arc CRLB skipped a draw: $(sprint(showerror, err))")
            end
        end
        covariances = ensemble_crlb(bounds)
        crlb = crlb_position(covariances, 1:3)
        times_h = case.times ./ 3600
        warm = draw_orbit(MersenneTwister(seed + 8_000_000 + index), case)
        meta = (;
            case = case.name,
            setting = "gap=$(plan.gap_h)h",
            ratio = NaN,
            requested_gap_h = plan.gap_h,
            actual_gap_h = case.actual_gap / 3600,
        )
        for spec in specs
            println("  warmup $(spec.name)")
            flush(stdout)
            run_orbit(spec, case, [warm])
            println("  run $(spec.name)")
            flush(stdout)
            raw = run_orbit(spec, case, draws)
            summary = summarize_filter(
                spec, raw, times_h, 1:3, 4:6, 6, crlb, case.gap_epoch, case.final_epoch, meta,
            )
            push!(rows, summary)
            @printf(
                "    pos RMSE %s   NEES %s   factor fail %.3f   diverge %.3f   reacquire %.3f   s/step %.3e\n",
                fmt(summary.rmse), fmt(summary.nees), summary.factor_fail, summary.diverge,
                summary.reacquire, summary.runtime,
            )
            flush(stdout)
        end
    end
    return rows
end

function ill_panel_series(rows, ratio)
    wanted = [
        ("CUT-4", "Float64"),
        ("SR-CUT-4", "Float64"),
        ("CUT-6", "Float64"),
        ("SR-CUT-6", "Float64"),
        ("CUT-4", "Float32"),
        ("SR-CUT-4", "Float32"),
        ("CUT-6", "Float32"),
        ("SR-CUT-6", "Float32"),
        ("SR-CKF", "Float64"),
        ("UKF", "Float64"),
        ("SR-UKF", "Float64"),
    ]
    series = Summary[]
    for (name, precision) in wanted
        row = find_row(rows; ratio = ratio, name = name, precision = precision)
        row === nothing || push!(series, row)
    end
    return series
end

function write_results(path, ill_rows, arc_rows, cli, ratio, gap)
    open(path, "w") do io
        println(io, "# Square-root CUT benchmarks")
        println(io)
        println(io, "Monte Carlo size M = $(cli.M). Seed = $(cli.seed).")
        println(io, "Every filter in a setting uses the same draws. Float32 uses those Float64 draws and rounds inside the filter.")
        println(io, "The posterior CRLB is the Tichavský recursion on each Monte Carlo truth, averaged as a covariance.")
        println(io, "Factor failure means the covariance update is not positive definite, or a square-root QR or downdate threw.")
        println(io, "Divergence is a factor failure or a position error above the threshold (1e5 km, or 100 times the position prior, on the ill-conditioned case; 5000 km on the long arc).")
        println(io, "Reacquisition means the run finished and the final position error is under 25 km or under half the error at the end of the coast.")
        println(io, "Runtime is wall time per scheduled filter step after one untimed warmup, averaged over runs.")
        println(io, "RMSE and NEES skip non-finite samples, so a run that fails mid-arc still contributes the epochs before the failure. The failure rate is the companion column.")
        println(io, "The linear-Gaussian unit test matches SR-CUT to full-covariance CUT within 100 ε(Float64).")
        println(io)
        if !isempty(ill_rows)
            println(io, "## Ill-conditioned case")
            println(io)
            println(io, "State `[x, y, vx, vy, β]` in km, km/s, and 1/km. Two ranges, each with σ = 1 cm.")
            println(io, "Ratio is the position prior standard deviation divided by that range standard deviation.")
            println(io, "CUT and SR-CUT of orders 4 and 6, and SR-CKF, run in Float64 and Float32. UKF and SR-UKF are Float64 baselines.")
            println(io, "The time-history figure uses ratio $(ratio), the setting with the largest CUT minus SR-CUT factor-failure gap over orders 4 and 6 and both precisions (gap = $(fmt(gap))).")
            println(io)
            println(io, markdown_illcond(ill_rows, cli.M))
            println(io)
        end
        if !isempty(arc_rows)
            println(io, "## Long-arc angles-only case")
            println(io)
            println(io, "LEO is a 400 km circular orbit with J2 and drag (B = $(B_LEO) m²/kg).")
            println(io, "HEO is a 500 km by 25000 km orbit with the same force model (B = $(B_HEO) m²/kg).")
            println(io, "The discrete map is fixed-step RK4 with a 60 s step, shared by the truth and every filter. The posterior CRLB differentiates that same map along each Monte Carlo truth.")
            println(io, "Measurements are the east and north components of the line-of-sight unit vector, σ = $(ANGLE_SIGMA) rad.")
            println(io, "The coast is the next visibility window at least as long as the requested gap. CUT-6 is not in this sweep: a 6-state CUT6 set has 137 points and the closed-form ill-conditioned sweep already compares order 6. This sweep uses CUT-4 (76 points), SR-CUT-4, SR-CKF (12 points), UKF, and SR-UKF.")
            println(io, "The RMSE figure caps the axis at 1e6 km when a series exceeds that value, so one divergent sample does not hide the coast and the PCRLB.")
            println(io)
            println(io, markdown_longarc(arc_rows))
            println(io)
        end
    end
    return path
end

function main()
    cli = parse_cli(ARGS)
    mkpath(OUTPUT_DIR)
    println("M = $(cli.M)  seed = $(cli.seed)")
    check_propagators()
    ill_rows = Summary[]
    arc_rows = Summary[]
    ratio = ILL_RATIOS[end]
    gap = NaN
    if cli.illcond
        ill_rows = run_illcond_study(cli.M, cli.seed)
        write_illcond_table(joinpath(OUTPUT_DIR, "illcond_table.csv"), ill_rows)
        ratio, gap = illcond_plot_ratio(ill_rows)
        series = ill_panel_series(ill_rows, ratio)
        panel = (;
            title = "σ_pos/σ_range = $(ratio)",
            xlabel = "time (s)",
            series = series,
            dof = 5,
            M = cli.M,
        )
        save_rmse_figure(joinpath(OUTPUT_DIR, "illcond_rmse.png"), [panel], "Ill-conditioned range tracking")
        save_nees_figure(joinpath(OUTPUT_DIR, "illcond_nees.png"), [panel], "Ill-conditioned NEES")
        save_cond_figure(joinpath(OUTPUT_DIR, "illcond_cond.png"), [panel], "Ill-conditioned median cond(P)")
        copy_figure(joinpath(OUTPUT_DIR, "illcond_rmse.png"), "srcut_illcond_rmse.png")
        copy_figure(joinpath(OUTPUT_DIR, "illcond_nees.png"), "srcut_illcond_nees.png")
        copy_figure(joinpath(OUTPUT_DIR, "illcond_cond.png"), "srcut_illcond_cond.png")
        println("ill-conditioned figure ratio = $ratio  failure gap = $gap")
    end
    if cli.longarc
        masks = prepare_masks()
        arc_rows = run_longarc_study(cli.M, cli.seed, masks)
        write_longarc_table(joinpath(OUTPUT_DIR, "longarc_table.csv"), arc_rows)
        panels = []
        for variant in ("LEO", "HEO")
            subset = filter(row -> row.case == variant, arc_rows)
            isempty(subset) && continue
            actual = maximum(row.actual_gap_h for row in subset)
            series = filter(row -> row.actual_gap_h == actual, subset)
            push!(
                panels,
                (;
                    title = "$variant coast $(round(actual; digits = 2)) h",
                    xlabel = "time (h)",
                    series = series,
                    dof = 6,
                    M = cli.M,
                ),
            )
        end
        isempty(panels) && error("long-arc study produced no scheduled passes")
        save_rmse_figure(joinpath(OUTPUT_DIR, "longarc_rmse.png"), panels, "Angles-only long-arc RMSE")
        save_nees_figure(joinpath(OUTPUT_DIR, "longarc_nees.png"), panels, "Angles-only long-arc NEES")
        save_cond_figure(joinpath(OUTPUT_DIR, "longarc_cond.png"), panels, "Angles-only long-arc cond(P)")
        copy_figure(joinpath(OUTPUT_DIR, "longarc_rmse.png"), "srcut_longarc_rmse.png")
        copy_figure(joinpath(OUTPUT_DIR, "longarc_nees.png"), "srcut_longarc_nees.png")
        copy_figure(joinpath(OUTPUT_DIR, "longarc_cond.png"), "srcut_longarc_cond.png")
    end
    results = joinpath(@__DIR__, "srcut_results.md")
    write_results(results, ill_rows, arc_rows, cli, ratio, gap)
    println("wrote $results")
    return nothing
end

main()
