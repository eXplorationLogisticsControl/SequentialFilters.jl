"""
CairoMakie figure of estimation error and ±3σ bounds.

`err[k, i]` is component `i` of (estimate − truth) at sample `k`.
`sig[k, i]` is `sqrt(P[i, i])` at that time. The band is drawn at `±3 sig`.
"""

using CairoMakie
using Printf

function push_error!(t_hist, err_hist, sig_hist, t, xhat, xtrue, P)
    push!(t_hist, t)
    push!(err_hist, collect(Float64, xhat) .- collect(Float64, xtrue))
    push!(sig_hist, sqrt.(max.(diag(P), 0.0)))
    return nothing
end

function save_error_figure(path, time_minutes, err, sig, labels, title)
    CairoMakie.activate!(type = "png")
    n = length(labels)
    ncols = 3
    nrows = cld(n, ncols)
    fig = Figure(size = (1000, 320 * nrows))
    Label(fig[0, 1:ncols], title; fontsize = 16)
    axes = Matrix{Axis}(undef, nrows, ncols)
    for i in 1:n
        row, col = fldmod1(i, ncols)
        axes[row, col] = Axis(
            fig[row, col];
            ylabel = labels[i],
            xlabel = row == nrows ? "time (min)" : "",
        )
        σ3 = 3 .* sig[:, i]
        band!(axes[row, col], time_minutes, -σ3, σ3; color = (:steelblue, 0.35))
        lines!(axes[row, col], time_minutes, err[:, i]; color = :black, linewidth = 1.2)
    end
    linkxaxes!(axes...)
    if nrows > 1
        hidexdecorations!.(axes[1:(nrows - 1), :]; grid = false, ticks = false)
    end
    Legend(
        fig[nrows + 1, 1:ncols],
        [PolyElement(color = (:steelblue, 0.35)), LineElement(color = :black, linewidth = 1.2)],
        ["±3σ  (±3 √P[i,i])", "error (estimate − truth)"];
        orientation = :horizontal,
        tellheight = true,
        framevisible = false,
    )
    save(path, fig)
    return abspath(path)
end

function print_twobody_summary(name, err_hist, outfile)
    # History is initial, then (predict, update) pairs. Odd entries from 3 on are posteriors.
    posterior = err_hist[3:2:end]
    pos = [norm(e[1:3]) for e in posterior]
    vel = [norm(e[4:6]) for e in posterior]
    pos_rmse = sqrt(sum(abs2, pos) / length(pos))
    vel_rmse = sqrt(sum(abs2, vel) / length(vel))
    println(name)
    println("sign convention: error = estimate - truth")
    println("bands: ±3 * sqrt(P[i, i]) about zero")
    println("RMSE uses the posterior (after each update)")
    @printf("position RMSE: %.6f km\n", pos_rmse)
    @printf("velocity RMSE: %.6e km/s\n", vel_rmse)
    @printf("final position error: %.6f km\n", pos[end])
    @printf("final velocity error: %.6e km/s\n", vel[end])
    println("saved $(abspath(outfile))")
    return nothing
end
