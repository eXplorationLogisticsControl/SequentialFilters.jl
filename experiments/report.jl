"""
Aggregate a Monte Carlo batch with the package diagnosis functions and write
the table and CairoMakie figures.

Time-averaged RMSE uses the posterior at each measurement instant
(`errors[:, :, 2:end]`). Plots of RMSE, NEES, and the Cramér–Rao bound also
include the initial time. NIS is defined only at measurement times.
"""

const FILTER_STYLE = Dict(
    "EKF" => (color = :black, linestyle = :solid),
    "SR-EKF" => (color = :black, linestyle = :dash),
    "UKF" => (color = :royalblue, linestyle = :solid),
    "SR-UKF" => (color = :royalblue, linestyle = :dash),
    "CUT" => (color = :seagreen, linestyle = :solid),
    "SR-CUT" => (color = :seagreen, linestyle = :dash),
)

struct SummaryRow
    name::String
    position_rmse::Float64
    velocity_rmse::Float64
    divergence_rate::Float64
    runtime::Float64
end

function summarize(scenario::Scenario, results)
    posterior = 2:scenario.n_steps+1
    rows = SummaryRow[]
    for result in results
        posterior_errors = @view result.errors[:, :, posterior]
        averages = SequentialFilters.time_averaged_rmse(posterior_errors, scenario.blocks)
        push!(
            rows,
            SummaryRow(
                result.name,
                averages.position,
                averages.velocity,
                SequentialFilters.divergence_rate(result.diverged),
                result.runtime,
            ),
        )
    end
    return rows
end

function format_summary(rows)
    lines = String[]
    push!(lines, "filter    position RMSE (km)    velocity RMSE (km/s)    divergence rate    runtime (s)")
    for row in rows
        push!(
            lines,
            @sprintf(
                "%-8s %18.6f %24.6e %18.4f %14.4f",
                row.name,
                row.position_rmse,
                row.velocity_rmse,
                row.divergence_rate,
                row.runtime,
            ),
        )
    end
    return join(lines, "\n")
end

function write_summary_csv(path, rows)
    open(path, "w") do io
        println(io, "filter,position_rmse_km,velocity_rmse_km_s,divergence_rate,runtime_s")
        for row in rows
            println(
                io,
                row.name,
                ",",
                row.position_rmse,
                ",",
                row.velocity_rmse,
                ",",
                row.divergence_rate,
                ",",
                row.runtime,
            )
        end
    end
    return path
end

function _finite_innovation(nu)
    last = 0
    for k in 1:size(nu, 2)
        all(isfinite, @view nu[:, k]) || break
        last = k
    end
    return last == 0 ? nu[:, 1:0] : nu[:, 1:last]
end

"""Mean of per-run innovation autocorrelations. Lag 0 is the first entry."""
function mean_autocorrelation(innovations::AbstractArray{<:Real,3}, max_lag::Integer)
    last = Int(max_lag)
    acc = zeros(Float64, last + 1)
    count = zeros(Int, last + 1)
    for i in 1:size(innovations, 2)
        series = _finite_innovation(@view innovations[:, i, :])
        N = size(series, 2)
        N >= 1 || continue
        ρ = SequentialFilters.innovation_autocorrelation(series; max_lag = last)
        for ell in 0:min(last, N - 1)
            acc[ell + 1] += ρ[ell + 1]
            count[ell + 1] += 1
        end
    end
    average = similar(acc)
    for i in eachindex(acc)
        average[i] = count[i] == 0 ? NaN : acc[i] / count[i]
    end
    return average
end

function _style(name)
    return get(FILTER_STYLE, name, (color = :gray, linestyle = :solid))
end

function _legend_elements(names; bound::Bool = false, crlb::Bool = false)
    elements = []
    labels = String[]
    for name in names
        style = _style(name)
        push!(elements, LineElement(color = style.color, linestyle = style.linestyle, linewidth = 1.8))
        push!(labels, name)
    end
    if crlb
        push!(elements, LineElement(color = :gray28, linestyle = :dot, linewidth = 2.2))
        push!(labels, "PCRLB")
    end
    if bound
        push!(elements, PolyElement(color = (:gray, 0.28)))
        push!(labels, "95% χ² band")
    end
    return elements, labels
end

function save_rmse_figure(path, scenario, results, position_bound, velocity_bound)
    CairoMakie.activate!(type = "png")
    minutes = scenario.times ./ 60
    fig = Figure(size = (980, 760))
    position_axis = Axis(fig[1, 1]; ylabel = "position RMSE (km)", xlabel = "time (min)")
    velocity_axis = Axis(fig[2, 1]; ylabel = "velocity RMSE (km/s)", xlabel = "time (min)")
    for result in results
        style = _style(result.name)
        position = SequentialFilters.block_rmse(result.errors, scenario.blocks.position)
        velocity = SequentialFilters.block_rmse(result.errors, scenario.blocks.velocity)
        lines!(position_axis, minutes, position; color = style.color, linestyle = style.linestyle, linewidth = 1.8)
        lines!(velocity_axis, minutes, velocity; color = style.color, linestyle = style.linestyle, linewidth = 1.8)
    end
    lines!(position_axis, minutes, position_bound; color = :gray28, linestyle = :dot, linewidth = 2.2)
    lines!(velocity_axis, minutes, velocity_bound; color = :gray28, linestyle = :dot, linewidth = 2.2)
    elements, labels = _legend_elements([result.name for result in results]; crlb = true)
    Legend(fig[1:2, 2], elements, labels; framevisible = false, tellheight = false)
    Label(fig[0, 1:2], "Two-body RMSE and posterior CRLB"; fontsize = 16)
    save(path, fig)
    return path
end

function save_nees_figure(path, scenario, results, M)
    CairoMakie.activate!(type = "png")
    minutes = scenario.times ./ 60
    lower, upper = mean_chi2_interval(scenario.nx, M)
    fig = Figure(size = (980, 520))
    axis = Axis(fig[1, 1]; ylabel = "average NEES", xlabel = "time (min)")
    band!(axis, minutes, fill(lower, length(minutes)), fill(upper, length(minutes)); color = (:gray, 0.28))
    for result in results
        style = _style(result.name)
        lines!(
            axis,
            minutes,
            SequentialFilters.averaged_nees(result.nees);
            color = style.color,
            linestyle = style.linestyle,
            linewidth = 1.8,
        )
    end
    elements, labels = _legend_elements([result.name for result in results]; bound = true)
    Legend(fig[1, 2], elements, labels; framevisible = false)
    Label(fig[0, 1:2], "Average NEES with a 95% χ²($(M)×$(scenario.nx)) band"; fontsize = 16)
    save(path, fig)
    return path
end

function save_nis_figure(path, scenario, results, M)
    CairoMakie.activate!(type = "png")
    minutes = scenario.times[2:end] ./ 60
    lower, upper = mean_chi2_interval(scenario.ny, M)
    fig = Figure(size = (980, 520))
    axis = Axis(fig[1, 1]; ylabel = "average NIS", xlabel = "time (min)")
    band!(axis, minutes, fill(lower, length(minutes)), fill(upper, length(minutes)); color = (:gray, 0.28))
    for result in results
        style = _style(result.name)
        lines!(
            axis,
            minutes,
            SequentialFilters.averaged_nis(result.nis);
            color = style.color,
            linestyle = style.linestyle,
            linewidth = 1.8,
        )
    end
    elements, labels = _legend_elements([result.name for result in results]; bound = true)
    Legend(fig[1, 2], elements, labels; framevisible = false)
    Label(fig[0, 1:2], "Average NIS with a 95% χ²($(M)×$(scenario.ny)) band"; fontsize = 16)
    save(path, fig)
    return path
end

function save_autocorrelation_figure(path, scenario, results)
    CairoMakie.activate!(type = "png")
    max_lag = min(10, scenario.n_steps - 1)
    lags = 0:max_lag
    bound = SequentialFilters.whiteness_bound(scenario.n_steps)
    fig = Figure(size = (980, 520))
    axis = Axis(fig[1, 1]; ylabel = "innovation autocorrelation", xlabel = "lag")
    hlines!(axis, [bound, -bound]; color = :gray28, linestyle = :dot, linewidth = 1.6)
    for result in results
        style = _style(result.name)
        ρ = mean_autocorrelation(result.innovations, max_lag)
        lines!(axis, lags, ρ; color = style.color, linestyle = style.linestyle, linewidth = 1.8)
    end
    elements, labels = _legend_elements([result.name for result in results])
    push!(elements, LineElement(color = :gray28, linestyle = :dot, linewidth = 1.6))
    push!(labels, "±1.96/√N")
    Legend(fig[1, 2], elements, labels; framevisible = false)
    Label(fig[0, 1:2], "Innovation autocorrelation"; fontsize = 16)
    save(path, fig)
    return path
end

function write_sensitivity_csv(path, records)
    open(path, "w") do io
        println(io, "measurement_noise_scale,filter,position_rmse_km,velocity_rmse_km_s,divergence_rate,runtime_s,pcrlb_position_km,pcrlb_velocity_km_s")
        for record in records
            println(
                io,
                record.scale,
                ",",
                record.name,
                ",",
                record.position_rmse,
                ",",
                record.velocity_rmse,
                ",",
                record.divergence_rate,
                ",",
                record.runtime,
                ",",
                record.pcrlb_position,
                ",",
                record.pcrlb_velocity,
            )
        end
    end
    return path
end
