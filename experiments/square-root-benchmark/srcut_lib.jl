"""
Ill-conditioned tracking and long-arc angles-only benchmarks.

The ill-conditioned case is a planar drag model with mixed units: position in
km, velocity in km/s, and a weakly observed drag coefficient. Range
measurements are 1 cm. The prior position standard deviation is swept so the
prior-to-measurement ratio covers several orders of magnitude. Each ratio is
run in Float64 and Float32. Full-covariance filters fail when the updated
covariance is not positive definite. Square-root filters fail only when their
own QR or downdate throws.

The long-arc case is angles-only orbit determination. LEO dynamics include J2
and drag. The second variant is a high-Earth orbit with the same force model.
Passes are separated by a coast of the requested length, or the next
visibility window after that coast. The posterior CRLB is the Tichavský
recursion on each Monte Carlo truth, then the covariance bounds are averaged.
A single noise-free reference is not used.
"""

using CairoMakie
using ForwardDiff
using LinearAlgebra
using Printf
using Random
using Statistics

import SequentialFilters

include(joinpath(@__DIR__, "metrics.jl"))

const MEDIA_DIR = "/cursor/stores/bc-01a16e62-1ec1-7d18-bb7e-3bb5abaadfe9/media"

# --- orbit model ----------------------------------------------------------------

const MU_EARTH = 3.986004418e5
const RE_EARTH = 6378.137
const J2_EARTH = 1.08262668e-3
const OMEGA_EARTH = 7.292115146706979e-5
const STATION_LAT = deg2rad(28.5)
const STATION_LON0 = 0.0
const RHO_400 = 3.0e-12
const H_REF_KM = 400.0
const H_SCALE_KM = 60.0
const RK_STEP = 60.0
const SIGMA_ACCEL = 1.0e-9
const ANGLE_SIGMA = 2.0e-4
const B_LEO = 0.022
const B_HEO = 0.015

mutable struct RKWork
    x::Vector{Float64}
    k1::Vector{Float64}
    k2::Vector{Float64}
    k3::Vector{Float64}
    k4::Vector{Float64}
    tmp::Vector{Float64}
end

const RK_WORK = Ref{RKWork}()

function rk_work()
    if !isassigned(RK_WORK)
        RK_WORK[] = RKWork(
            zeros(6), zeros(6), zeros(6), zeros(6), zeros(6), zeros(6),
        )
    end
    return RK_WORK[]
end

"""J2 plus exponential drag. The atmosphere does not rotate. `B` is Cd*A/m in m²/kg."""
function accel_tuple(x, y, z, vx, vy, vz, B)
    r2 = x * x + y * y + z * z
    r1 = sqrt(r2)
    inv_r3 = MU_EARTH / (r2 * r1)
    ax = -inv_r3 * x
    ay = -inv_r3 * y
    az = -inv_r3 * z
    factor = 1.5 * J2_EARTH * MU_EARTH * RE_EARTH * RE_EARTH / (r2 * r2 * r1)
    z2 = z * z
    five = 5 * z2 / r2
    ax += factor * x * (five - 1)
    ay += factor * y * (five - 1)
    az += factor * z * (five - 3)
    alt = r1 - RE_EARTH
    ρ = RHO_400 * exp(-(alt - H_REF_KM) / H_SCALE_KM)
    speed_mps = 1000 * sqrt(vx * vx + vy * vy + vz * vz)
    drag = -0.5 * B * ρ * speed_mps
    ax += drag * vx
    ay += drag * vy
    az += drag * vz
    return vx, vy, vz, ax, ay, az
end

function deriv!(du, u, B)
    dx, dy, dz, ax, ay, az = accel_tuple(u[1], u[2], u[3], u[4], u[5], u[6], B)
    du[1] = dx
    du[2] = dy
    du[3] = dz
    du[4] = ax
    du[5] = ay
    du[6] = az
    return du
end

function deriv_vec(u, B)
    dx, dy, dz, ax, ay, az = accel_tuple(u[1], u[2], u[3], u[4], u[5], u[6], B)
    return [dx, dy, dz, ax, ay, az]
end

function rk4!(work, h, B)
    deriv!(work.k1, work.x, B)
    @. work.tmp = work.x + (h / 2) * work.k1
    deriv!(work.k2, work.tmp, B)
    @. work.tmp = work.x + (h / 2) * work.k2
    deriv!(work.k3, work.tmp, B)
    @. work.tmp = work.x + h * work.k3
    deriv!(work.k4, work.tmp, B)
    @. work.x = work.x + (h / 6) * (work.k1 + 2 * work.k2 + 2 * work.k3 + work.k4)
    return work.x
end

function n_substeps(dt)
    return max(1, ceil(Int, abs(Float64(dt)) / RK_STEP))
end

function propagate_fast(x::AbstractVector{Float64}, dt::Float64, B::Float64)
    work = rk_work()
    copyto!(work.x, x)
    n = n_substeps(dt)
    h = dt / n
    for _ in 1:n
        rk4!(work, h, B)
    end
    return copy(work.x)
end

function propagate_generic(x::AbstractVector, dt::Real, B::Real)
    n = n_substeps(dt)
    h = dt / n
    z = copy(x)
    for _ in 1:n
        k1 = deriv_vec(z, B)
        k2 = deriv_vec(z .+ (h / 2) .* k1, B)
        k3 = deriv_vec(z .+ (h / 2) .* k2, B)
        k4 = deriv_vec(z .+ h .* k3, B)
        z = z .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    return z
end

function propagate(x::AbstractVector{Float64}, dt::Real, B::Real)
    return propagate_fast(x, Float64(dt), Float64(B))
end

function propagate(x::AbstractVector, dt::Real, B::Real)
    return propagate_generic(x, dt, B)
end

function local_frame(t)
    λ = STATION_LON0 + OMEGA_EARTH * t
    sλ, cλ = sincos(λ)
    sφ, cφ = sincos(STATION_LAT)
    east = (-sλ, cλ, 0.0)
    north = (-sφ * cλ, -sφ * sλ, cφ)
    zenith = (cφ * cλ, cφ * sλ, sφ)
    rs = RE_EARTH .* zenith
    return rs, east, north, zenith
end

function los_measurement(t, x)
    rs, east, north, _ = local_frame(t)
    ρx = x[1] - rs[1]
    ρy = x[2] - rs[2]
    ρz = x[3] - rs[3]
    ρ = sqrt(ρx * ρx + ρy * ρy + ρz * ρz)
    easting = east[1] * ρx + east[2] * ρy + east[3] * ρz
    northing = north[1] * ρx + north[2] * ρy + north[3] * ρz
    return [easting / ρ, northing / ρ]
end

function elevation_angle(t, x)
    rs, _, _, zenith = local_frame(t)
    ρx = x[1] - rs[1]
    ρy = x[2] - rs[2]
    ρz = x[3] - rs[3]
    ρ = sqrt(ρx * ρx + ρy * ρy + ρz * ρz)
    uz = (zenith[1] * ρx + zenith[2] * ρy + zenith[3] * ρz) / ρ
    return asin(clamp(uz, -1.0, 1.0))
end

function discrete_process_covariance(dt, σa = SIGMA_ACCEL)
    q = σa * σa
    a = q * dt^3 / 3
    b = q * dt^2 / 2
    c = q * dt
    Q = zeros(6, 6)
    for i in 1:3
        Q[i, i] = a
        Q[i, i + 3] = b
        Q[i + 3, i] = b
        Q[i + 3, i + 3] = c
    end
    return Q
end

function add_process_noise!(x, dt, rng, σa = SIGMA_ACCEL)
    q = σa * σa
    a = q * dt^3 / 3
    b = q * dt^2 / 2
    L11 = sqrt(a)
    L21 = b / L11
    L22 = 0.5 * σa * sqrt(dt)
    for i in 1:3
        z1 = randn(rng)
        z2 = randn(rng)
        x[i] += L11 * z1
        x[i + 3] += L21 * z1 + L22 * z2
    end
    return x
end

function circular_state(altitude, inclination)
    a = RE_EARTH + altitude
    lat = STATION_LAT
    r = a .* (cos(lat), 0.0, sin(lat))
    speed = sqrt(MU_EARTH / a)
    sine_heading = cos(inclination) / cos(lat)
    abs(sine_heading) <= 1 || error("inclination is below the station latitude")
    heading = asin(sine_heading)
    east = (0.0, 1.0, 0.0)
    north = (-sin(lat), 0.0, cos(lat))
    v = speed .* (sin(heading) .* east .+ cos(heading) .* north)
    return vcat(collect(r), collect(v))
end

function heo_state()
    rp = RE_EARTH + 500.0
    ra = RE_EARTH + 25000.0
    a = 0.5 * (rp + ra)
    lat = STATION_LAT
    r = rp .* (cos(lat), 0.0, sin(lat))
    speed = sqrt(MU_EARTH * (2 / rp - 1 / a))
    inclination = deg2rad(63.4)
    heading = asin(clamp(cos(inclination) / cos(lat), -1.0, 1.0))
    east = (0.0, 1.0, 0.0)
    north = (-sin(lat), 0.0, cos(lat))
    v = speed .* (sin(heading) .* east .+ cos(heading) .* north)
    return vcat(collect(r), collect(v))
end

function describe_orbit(label, x)
    r = x[1:3]
    v = x[4:6]
    rn = norm(r)
    vn = norm(v)
    h = cross(r, v)
    inc = acosd(h[3] / norm(h))
    energy = vn^2 / 2 - MU_EARTH / rn
    a = -MU_EARTH / (2 * energy)
    period_min = 2π * sqrt(abs(a)^3 / MU_EARTH) / 60
    el = rad2deg(elevation_angle(0.0, x))
    @printf(
        "%s  r = %.1f km  v = %.4f km/s  inc = %.2f deg  a = %.1f km  period = %.1f min  el(0) = %.1f deg\n",
        label, rn, vn, inc, a, period_min, el,
    )
    return nothing
end

function visibility_mask(x0, B, horizon, dt = 30.0)
    n = floor(Int, horizon / dt) + 1
    times = Vector{Float64}(undef, n)
    elevs = Vector{Float64}(undef, n)
    x = copy(x0)
    t = 0.0
    for i in 1:n
        times[i] = t
        elevs[i] = elevation_angle(t, x)
        i == n && break
        x = propagate(x, dt, B)
        t += dt
    end
    return times, elevs
end

function sample_forward(times, elevs, idxs, meas_dt, n_meas, elmin)
    picked = Int[]
    isempty(idxs) && return picked
    next_t = times[idxs[1]]
    for i in idxs
        times[i] + 1e-8 < next_t && continue
        elevs[i] < elmin && continue
        push!(picked, i)
        next_t = times[i] + meas_dt
        length(picked) >= n_meas && break
    end
    return picked
end

struct OrbitStep
    t0::Float64
    t1::Float64
    dt::Float64
    record_prediction::Bool
end

struct OrbitCase
    name::String
    x0::Vector{Float64}
    B::Float64
    P0::Matrix{Float64}
    R::Matrix{Float64}
    steps::Vector{OrbitStep}
    requested_gap::Float64
    actual_gap::Float64
    times::Vector{Float64}
    gap_epoch::Int
    final_epoch::Int
end

function build_orbit_case(mask, requested_gap; σ_r, σ_v, elmin = deg2rad(10), meas_dt = 30.0, n_meas = 8)
    times, elevs = mask.times, mask.elevs
    above = findall(e -> e >= elmin, elevs)
    isempty(above) && return nothing
    windows = Vector{UnitRange{Int}}()
    start = above[1]
    previous = above[1]
    for i in above[2:end]
        if i == previous + 1
            previous = i
        else
            push!(windows, start:previous)
            start = i
            previous = i
        end
    end
    push!(windows, start:previous)
    pass1_idx = sample_forward(times, elevs, collect(windows[1]), meas_dt, n_meas + 1, elmin)
    t1 = [times[i] for i in pass1_idx if times[i] >= meas_dt - 1e-6]
    length(t1) >= 4 || return nothing
    t_end = t1[end]
    target = t_end + requested_gap
    pass2_idx = Int[]
    for window in windows
        idxs = [i for i in window if times[i] >= target - 1e-6]
        isempty(idxs) && continue
        pass2_idx = sample_forward(times, elevs, idxs, meas_dt, n_meas, elmin)
        length(pass2_idx) >= 4 && break
        pass2_idx = Int[]
    end
    length(pass2_idx) >= 4 || return nothing
    t2 = times[pass2_idx]
    meas = vcat(t1, t2)
    steps = OrbitStep[]
    t = 0.0
    for tm in meas
        push!(steps, OrbitStep(t, tm, tm - t, tm == t2[1]))
        t = tm
    end
    epoch_times = Float64[0.0]
    gap_epoch = 0
    for step in steps
        if step.record_prediction
            push!(epoch_times, step.t1)
            gap_epoch = length(epoch_times)
        end
        push!(epoch_times, step.t1)
    end
    P0 = Matrix(Diagonal([fill(σ_r^2, 3); fill(σ_v^2, 3)]))
    R = Matrix((ANGLE_SIGMA^2) * I(2))
    return OrbitCase(
        mask.name,
        mask.x0,
        mask.B,
        P0,
        R,
        steps,
        requested_gap,
        t2[1] - t_end,
        epoch_times,
        gap_epoch,
        length(epoch_times),
    )
end

# --- ill-conditioned model ------------------------------------------------------

const ILL_X0 = [8.0, 12.0, 0.04, -0.03, 1.0e-3]
const ILL_DT = 1.0
const ILL_STEPS = 20
const ILL_SIGMA_V = 5.0e-3
const ILL_SIGMA_BETA = 0.2
const ILL_SIGMA_RANGE = 1.0e-5
const ILL_RATIOS = (1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8)

function ill_step(x, dt)
    px, py, vx, vy, β = x[1], x[2], x[3], x[4], x[5]
    half = oftype(vx, 0.5)
    tiny = oftype(vx, 1.0e-12)
    speed = sqrt(vx * vx + vy * vy + tiny * tiny)
    ax = -β * speed * vx
    ay = -β * speed * vy
    dt2 = dt * dt
    return [
        px + vx * dt + half * ax * dt2,
        py + vy * dt + half * ay * dt2,
        vx + ax * dt,
        vy + ay * dt,
        β,
    ]
end

function ill_measurement(t, x)
    px, py = x[1], x[2]
    range1 = sqrt(px * px + py * py)
    dx = px - oftype(px, 30)
    dy = py - oftype(py, 5)
    range2 = sqrt(dx * dx + dy * dy)
    return [range1, range2]
end

function ill_process_covariance()
    return diagm([1.0e-8, 1.0e-8, 1.0e-12, 1.0e-12, 1.0e-10])
end

function ill_measurement_covariance()
    s2 = ILL_SIGMA_RANGE^2
    return diagm([s2, s2])
end

function ill_prior(σ_position)
    return diagm([σ_position^2, σ_position^2, ILL_SIGMA_V^2, ILL_SIGMA_V^2, ILL_SIGMA_BETA^2])
end

function ill_threshold(σ_position)
    return max(1.0e5, 100 * σ_position)
end

# --- filters --------------------------------------------------------------------

struct BenchFilter
    name::String
    precision::String
    order::Int
    n_sigma::Int
    covariance_form::Bool
    build::Function
end

function _count_cut(nx, order)
    return length(SequentialFilters.cut_points(nx; order = order)[2])
end

function bench_filters(nx; precisions = (Float64, Float32), orders = (4, 6), ukf::Bool = true)
    SF = SequentialFilters
    specs = BenchFilter[]
    for T in precisions
        precision = T == Float32 ? "Float32" : "Float64"
        for order in orders
            n_sigma = _count_cut(nx, order)
            push!(
                specs,
                BenchFilter(
                    "CUT-$order",
                    precision,
                    order,
                    n_sigma,
                    true,
                    (x, P) -> SF.ConjugateUnscentedKalmanFilter(zero(T), Vector{T}(x), Matrix{T}(P); order = order),
                ),
            )
            push!(
                specs,
                BenchFilter(
                    "SR-CUT-$order",
                    precision,
                    order,
                    n_sigma,
                    false,
                    (x, P) -> SF.SquareRootConjugateUnscentedKalmanFilter(
                        zero(T), Vector{T}(x), Matrix{T}(P); order = order,
                    ),
                ),
            )
        end
        push!(
            specs,
            BenchFilter(
                "SR-CKF",
                precision,
                0,
                2 * nx,
                false,
                (x, P) -> SF.SquareRootCubatureKalmanFilter(zero(T), Vector{T}(x), Matrix{T}(P)),
            ),
        )
    end
    if ukf
        push!(
            specs,
            BenchFilter(
                "UKF",
                "Float64",
                0,
                2 * nx + 1,
                true,
                (x, P) -> SF.UnscentedKalmanFilter(0.0, Vector{Float64}(x), Matrix{Float64}(P)),
            ),
        )
        push!(
            specs,
            BenchFilter(
                "SR-UKF",
                "Float64",
                0,
                2 * nx + 1,
                false,
                (x, P) -> SF.SquareRootUnscentedKalmanFilter(0.0, Vector{Float64}(x), Matrix{Float64}(P)),
            ),
        )
    end
    return specs
end

function filter_mean(filt)
    return filt.x
end

function filter_covariance(filt)
    return hasproperty(filt, :P) ? filt.P : SequentialFilters.covariance(filt)
end

# --- numerics and CRLB ----------------------------------------------------------

function symmetrize(A)
    return (A + A') ./ 2
end

function safe_cond(P)
    if !all(isfinite, P)
        return NaN
    end
    ev = eigvals(Symmetric(symmetrize(P)))
    amax = maximum(abs, ev)
    amin = minimum(abs, ev)
    if !(amin > 0) || !isfinite(amax)
        return Inf
    end
    return amax / amin
end

function not_positive_definite(P)
    all(isfinite, P) || return true
    return !issuccess(cholesky(Hermitian(symmetrize(P)); check = false))
end

function is_factor_exception(err)
    err isa PosDefException && return true
    err isa SingularException && return true
    if err isa ArgumentError
        msg = err.msg
        return occursin("positive definite", msg) ||
            occursin("positive semidefinite", msg) ||
            occursin("Cholesky", msg)
    end
    return false
end

function information_update(Ppred, H, R)
    predicted = symmetrize(Ppred)
    info = Matrix(Symmetric(predicted) \ I(size(predicted, 1)))
    info .+= Matrix(H)' * Matrix(R \ H)
    info = symmetrize(info)
    posterior = Matrix(Symmetric(info) \ I(size(info, 1)))
    return symmetrize(info), symmetrize(posterior)
end

function predict_covariance(J, F, Q)
    predicted = Q + F * (Symmetric(J) \ collect(F'))
    return symmetrize(predicted)
end

function mean_covariances(blocks)
    acc = zeros(size(blocks[1]))
    n = 0
    for block in blocks
        if all(isfinite, block)
            acc .+= block
            n += 1
        end
    end
    n > 0 || error("every posterior CRLB sample failed")
    return acc ./ n
end

function crlb_position(covariances, indices)
    bound = Vector{Float64}(undef, size(covariances, 3))
    for k in 1:length(bound)
        block = covariances[indices, indices, k]
        bound[k] = sqrt(max(tr(block), 0.0))
    end
    return bound
end

function ill_crlb(draw)
    nx = 5
    n_epoch = ILL_STEPS + 1
    covariances = Array{Float64}(undef, nx, nx, n_epoch)
    covariances[:, :, 1] = draw.P0
    J = symmetrize(Matrix(Symmetric(draw.P0) \ I(nx)))
    t = 0.0
    for k in 1:ILL_STEPS
        F = ForwardDiff.jacobian(z -> collect(ill_step(z, ILL_DT)), draw.truth[:, k])
        Ppred = predict_covariance(J, F, draw.Q)
        t += ILL_DT
        H = ForwardDiff.jacobian(z -> collect(ill_measurement(t, z)), draw.truth[:, k + 1])
        J, posterior = information_update(Ppred, H, draw.R)
        covariances[:, :, k + 1] = posterior
    end
    return covariances
end

"""Central difference of a Float64 map. The step is relative so kilometre and km/s states both move."""
function jacobian_fast(f, x; relative::Float64 = 1.0e-6)
    y0 = f(x)
    m = length(y0)
    n = length(x)
    J = Matrix{Float64}(undef, m, n)
    for i in 1:n
        step = relative * max(1.0, abs(x[i]))
        xp = copy(x)
        xm = copy(x)
        xp[i] += step
        xm[i] -= step
        fp = f(xp)
        fm = f(xm)
        for row in 1:m
            J[row, i] = (fp[row] - fm[row]) / (2 * step)
        end
    end
    return J
end

function orbit_crlb(case, draw)
    nx = 6
    n_epoch = length(case.times)
    covariances = Array{Float64}(undef, nx, nx, n_epoch)
    covariances[:, :, 1] = case.P0
    J = symmetrize(Matrix(Symmetric(case.P0) \ I(nx)))
    epoch = 1
    for (k, step) in enumerate(case.steps)
        # The bound uses the same discrete map the filter propagates, not a
        # separate noise-free reference. Finite differences stay on that map.
        F = jacobian_fast(z -> propagate(z, step.dt, case.B), draw.truth[:, k])
        Q = discrete_process_covariance(step.dt)
        Ppred = predict_covariance(J, F, Q)
        if step.record_prediction
            epoch += 1
            covariances[:, :, epoch] = Ppred
        end
        H = jacobian_fast(z -> los_measurement(step.t1, z), draw.truth[:, k + 1])
        J, posterior = information_update(Ppred, H, case.R)
        epoch += 1
        covariances[:, :, epoch] = posterior
    end
    epoch == n_epoch || error("CRLB produced $epoch epochs, expected $n_epoch")
    return covariances
end

# --- Monte Carlo ----------------------------------------------------------------

struct RawRun
    errors::Array{Float64,3}
    nees::Matrix{Float64}
    conds::Matrix{Float64}
    factor_failed::Vector{Bool}
    diverged::Vector{Bool}
    reacquired::Vector{Bool}
    runtime_per_step::Float64
end

function store_epoch!(errors, nees_hist, conds, i, epoch, truth, xhat, P, pos)
    residual = SequentialFilters.estimation_error(truth, xhat)
    errors[:, i, epoch] = residual
    P64 = Matrix{Float64}(P)
    conds[i, epoch] = safe_cond(P64)
    try
        nees_hist[i, epoch] = SequentialFilters.nees(residual, P64)
    catch
        nees_hist[i, epoch] = NaN
    end
    finite = all(isfinite, residual[pos])
    return finite ? norm(residual[pos]) : NaN
end

function run_illcond(spec::BenchFilter, draws, σ_position)
    M = length(draws)
    nx = 5
    n_epoch = ILL_STEPS + 1
    errors = fill(NaN, nx, M, n_epoch)
    nees_hist = fill(NaN, M, n_epoch)
    conds = fill(NaN, M, n_epoch)
    factor_failed = falses(M)
    diverged = falses(M)
    reacquired = falses(M)
    Q = ill_process_covariance()
    R = ill_measurement_covariance()
    threshold = ill_threshold(σ_position)
    f = (t, x, dt) -> ill_step(x, dt)
    h = (t, x) -> ill_measurement(t, x)
    elapsed = 0.0
    for i in 1:M
        draw = draws[i]
        started = time_ns()
        local filt
        try
            filt = spec.build(draw.xhat0, draw.P0)
            xhat = filter_mean(filt)
            P = filter_covariance(filt)
            store_epoch!(errors, nees_hist, conds, i, 1, draw.truth[:, 1], xhat, P, 1:2)
            for k in 1:ILL_STEPS
                SequentialFilters.predict!(
                    filt,
                    SequentialFilters.DiscreteDynamicsModel(f; n = nx),
                    SequentialFilters.AdditiveProcessNoise(Q),
                    Float64(ILL_DT),
                )
                SequentialFilters.update!(filt, h, draw.ys[k], R)
                xhat = filter_mean(filt)
                P = filter_covariance(filt)
                if !(all(isfinite, xhat) && all(isfinite, P))
                    factor_failed[i] = true
                    diverged[i] = true
                    break
                end
                if spec.covariance_form && not_positive_definite(P)
                    factor_failed[i] = true
                    diverged[i] = true
                    break
                end
                pos = store_epoch!(errors, nees_hist, conds, i, k + 1, draw.truth[:, k + 1], xhat, P, 1:2)
                if SequentialFilters.is_diverged(view(errors, 1:2, i, k + 1); threshold = threshold, estimate = xhat, covariance = P)
                    diverged[i] = true
                end
                if !isfinite(pos)
                    diverged[i] = true
                    factor_failed[i] = true
                    break
                end
            end
        catch err
            if err isa InterruptException
                rethrow()
            end
            diverged[i] = true
            if is_factor_exception(err)
                factor_failed[i] = true
            else
                println("$(spec.name) $(spec.precision) unexpected: $(sprint(showerror, err))")
            end
        end
        elapsed += (time_ns() - started) / 1e9
    end
    return RawRun(errors, nees_hist, conds, factor_failed, diverged, reacquired, elapsed / (M * ILL_STEPS))
end

function run_orbit(spec::BenchFilter, case::OrbitCase, draws)
    M = length(draws)
    nx = 6
    n_epoch = length(case.times)
    n_steps = length(case.steps)
    errors = fill(NaN, nx, M, n_epoch)
    nees_hist = fill(NaN, M, n_epoch)
    conds = fill(NaN, M, n_epoch)
    factor_failed = falses(M)
    diverged = falses(M)
    reacquired = falses(M)
    f = (t, x, dt) -> propagate(collect(Float64, x), dt, case.B)
    h = (t, x) -> los_measurement(t, x)
    threshold = 5000.0
    elapsed = 0.0
    for i in 1:M
        draw = draws[i]
        started = time_ns()
        gap_error = NaN
        final_error = NaN
        local filt
        try
            filt = spec.build(draw.xhat0, case.P0)
            xhat = filter_mean(filt)
            P = filter_covariance(filt)
            store_epoch!(errors, nees_hist, conds, i, 1, draw.truth[:, 1], xhat, P, 1:3)
            epoch = 1
            stopped = false
            for (k, step) in enumerate(case.steps)
                Q = discrete_process_covariance(step.dt)
                SequentialFilters.predict!(
                    filt,
                    SequentialFilters.DiscreteDynamicsModel(f; n = nx),
                    SequentialFilters.AdditiveProcessNoise(Q),
                    Float64(step.dt),
                )
                xhat = filter_mean(filt)
                P = filter_covariance(filt)
                if !(all(isfinite, xhat) && all(isfinite, P)) ||
                    (spec.covariance_form && not_positive_definite(P))
                    factor_failed[i] = true
                    diverged[i] = true
                    stopped = true
                    break
                end
                if step.record_prediction
                    epoch += 1
                    gap_error = store_epoch!(
                        errors, nees_hist, conds, i, epoch, draw.truth[:, k + 1], xhat, P, 1:3,
                    )
                    if SequentialFilters.is_diverged(view(errors, 1:3, i, epoch); threshold = threshold, estimate = xhat, covariance = P)
                        diverged[i] = true
                    end
                end
                SequentialFilters.update!(filt, h, draw.ys[k], case.R)
                xhat = filter_mean(filt)
                P = filter_covariance(filt)
                if !(all(isfinite, xhat) && all(isfinite, P)) ||
                    (spec.covariance_form && not_positive_definite(P))
                    factor_failed[i] = true
                    diverged[i] = true
                    stopped = true
                    break
                end
                epoch += 1
                final_error = store_epoch!(
                    errors, nees_hist, conds, i, epoch, draw.truth[:, k + 1], xhat, P, 1:3,
                )
                if SequentialFilters.is_diverged(view(errors, 1:3, i, epoch); threshold = threshold, estimate = xhat, covariance = P)
                    diverged[i] = true
                end
            end
            if !stopped && isfinite(gap_error) && isfinite(final_error)
                reacquired[i] = final_error < 0.5 * gap_error || final_error < 25.0
            end
        catch err
            if err isa InterruptException
                rethrow()
            end
            diverged[i] = true
            if is_factor_exception(err)
                factor_failed[i] = true
            else
                println("$(spec.name) $(case.name) unexpected: $(sprint(showerror, err))")
            end
        end
        elapsed += (time_ns() - started) / 1e9
    end
    return RawRun(errors, nees_hist, conds, factor_failed, diverged, reacquired, elapsed / (M * n_steps))
end

function draw_illcond(rng, σ_position)
    P0 = ill_prior(σ_position)
    Q = ill_process_covariance()
    R = ill_measurement_covariance()
    scales0 = [σ_position, σ_position, ILL_SIGMA_V, ILL_SIGMA_V, ILL_SIGMA_BETA]
    scalesQ = sqrt.(diag(Q))
    xhat0 = ILL_X0 .+ scales0 .* randn(rng, 5)
    truth = Matrix{Float64}(undef, 5, ILL_STEPS + 1)
    truth[:, 1] = ILL_X0
    x = copy(ILL_X0)
    ys = Vector{Vector{Float64}}(undef, ILL_STEPS)
    t = 0.0
    for k in 1:ILL_STEPS
        x = collect(Float64, ill_step(x, ILL_DT)) .+ scalesQ .* randn(rng, 5)
        t += ILL_DT
        truth[:, k + 1] = x
        ys[k] = collect(Float64, ill_measurement(t, x)) .+ ILL_SIGMA_RANGE .* randn(rng, 2)
    end
    return (; truth, ys, xhat0, P0, Q, R)
end

function draw_orbit(rng, case::OrbitCase)
    L0 = Matrix(cholesky(Hermitian(case.P0)).L)
    Lr = Matrix(cholesky(Hermitian(case.R)).L)
    xhat0 = case.x0 .+ L0 * randn(rng, 6)
    n_steps = length(case.steps)
    truth = Matrix{Float64}(undef, 6, n_steps + 1)
    truth[:, 1] = case.x0
    x = copy(case.x0)
    ys = Vector{Vector{Float64}}(undef, n_steps)
    for (k, step) in enumerate(case.steps)
        x = propagate(x, step.dt, case.B)
        add_process_noise!(x, step.dt, rng)
        truth[:, k + 1] = x
        ys[k] = los_measurement(step.t1, x) .+ Lr * randn(rng, 2)
    end
    return (; truth, ys, xhat0)
end

function ensemble_crlb(draw_bounds)
    return mean_covariances(draw_bounds)
end

# --- summaries and figures ------------------------------------------------------

struct Summary
    case::String
    setting::String
    ratio::Float64
    requested_gap_h::Float64
    actual_gap_h::Float64
    name::String
    precision::String
    order::Int
    n_sigma::Int
    M::Int
    rmse::Float64
    extra_rmse::Float64
    nees::Float64
    nees_in_band::Float64
    factor_fail::Float64
    diverge::Float64
    reacquire::Float64
    runtime::Float64
    mean_log10_cond::Float64
    rmse_over_crlb::Float64
    gap_rmse::Float64
    final_rmse::Float64
    times::Vector{Float64}
    rmse_series::Vector{Float64}
    nees_series::Vector{Float64}
    cond_series::Vector{Float64}
    crlb_series::Vector{Float64}
end

function median_skipnan(values)
    kept = filter(!isnan, values)
    return isempty(kept) ? NaN : median(kept)
end

function nees_band_fraction(nees_hist, dof)
    hits = 0
    total = 0
    for j in 2:size(nees_hist, 2)
        finite = filter(isfinite, @view nees_hist[:, j])
        m = length(finite)
        m >= 1 || continue
        lower, upper = mean_chi2_interval(dof, m)
        total += 1
        μ = sum(finite) / m
        if lower <= μ <= upper
            hits += 1
        end
    end
    return total == 0 ? NaN : hits / total
end

function summarize_filter(spec, raw, times, pos, extra, dof, crlb, gap_epoch, final_epoch, meta)
    posterior = 2:length(times)
    rmse_series = SequentialFilters.block_rmse(raw.errors, pos)
    extra_series = SequentialFilters.block_rmse(raw.errors, extra)
    nees_series = SequentialFilters.averaged_nees(raw.nees)
    cond_series = [median_skipnan(@view raw.conds[:, j]) for j in 1:length(times)]
    rmse_avg = SequentialFilters.time_averaged_rmse(rmse_series[posterior])
    extra_avg = SequentialFilters.time_averaged_rmse(extra_series[posterior])
    nees_avg = SequentialFilters.time_averaged_rmse(nees_series[posterior])
    crlb_avg = SequentialFilters.time_averaged_rmse(crlb[posterior])
    logs = Float64[]
    for value in cond_series[posterior]
        if isfinite(value) && value > 0
            push!(logs, log10(value))
        end
    end
    mean_log = isempty(logs) ? NaN : sum(logs) / length(logs)
    gap_rmse = gap_epoch == 0 ? NaN : rmse_series[gap_epoch]
    final_rmse = rmse_series[final_epoch]
    reacquire = gap_epoch == 0 ? NaN : SequentialFilters.divergence_rate(raw.reacquired)
    return Summary(
        meta.case,
        meta.setting,
        meta.ratio,
        meta.requested_gap_h,
        meta.actual_gap_h,
        spec.name,
        spec.precision,
        spec.order,
        spec.n_sigma,
        size(raw.nees, 1),
        rmse_avg,
        extra_avg,
        nees_avg,
        nees_band_fraction(raw.nees, dof),
        SequentialFilters.divergence_rate(raw.factor_failed),
        SequentialFilters.divergence_rate(raw.diverged),
        reacquire,
        raw.runtime_per_step,
        mean_log,
        crlb_avg > 0 ? rmse_avg / crlb_avg : NaN,
        gap_rmse,
        final_rmse,
        times,
        rmse_series,
        nees_series,
        cond_series,
        crlb,
    )
end

const SERIES_STYLE = Dict(
    "UKF" => (:royalblue, :solid),
    "SR-UKF" => (:royalblue, :dash),
    "CUT-4" => (:seagreen, :solid),
    "SR-CUT-4" => (:seagreen, :dash),
    "CUT-6" => (:purple, :solid),
    "SR-CUT-6" => (:purple, :dash),
    "SR-CKF" => (:darkorange, :solid),
)

function series_style(summary::Summary)
    color, linestyle = get(SERIES_STYLE, summary.name, (:gray, :solid))
    if summary.precision == "Float32"
        linestyle = summary.name == "SR-CUT-4" || summary.name == "SR-CUT-6" || summary.name == "SR-CKF" ? :dashdot : :dot
    end
    return color, linestyle
end

function series_label(summary::Summary)
    return summary.precision == "Float64" && (summary.name == "UKF" || summary.name == "SR-UKF") ?
        summary.name : "$(summary.name) $(summary.precision)"
end

function _legend(labels, colors, linestyles; crlb::Bool = false, band::Bool = false)
    elements = []
    names = String[]
    for i in eachindex(labels)
        push!(elements, LineElement(color = colors[i], linestyle = linestyles[i], linewidth = 1.8))
        push!(names, labels[i])
    end
    if crlb
        push!(elements, LineElement(color = :red, linestyle = :solid, linewidth = 2.2))
        push!(names, "PCRLB")
    end
    if band
        push!(elements, PolyElement(color = (:gray, 0.28)))
        push!(names, "95% χ² band")
    end
    return elements, names
end

function save_rmse_figure(path, panels, title)
    CairoMakie.activate!(type = "png")
    n = length(panels)
    fig = Figure(size = (1040, 420 * n))
    for (i, panel) in enumerate(panels)
        axis = Axis(fig[i, 1]; xlabel = panel.xlabel, ylabel = "position RMSE", yscale = log10, title = panel.title)
        for summary in panel.series
            color, linestyle = series_style(summary)
            lines!(axis, summary.times, summary.rmse_series; color = color, linestyle = linestyle, linewidth = 1.8)
        end
        if !isempty(panel.series)
            lines!(axis, panel.series[1].times, panel.series[1].crlb_series; color = :red, linewidth = 2.2)
        end
        finite_vals = Float64[]
        for summary in panel.series
            for value in summary.rmse_series
                if isfinite(value) && value > 0
                    push!(finite_vals, value)
                end
            end
            for value in summary.crlb_series
                if isfinite(value) && value > 0
                    push!(finite_vals, value)
                end
            end
        end
        if !isempty(finite_vals) && maximum(finite_vals) > 1.0e6
            ylims!(axis, minimum(finite_vals) / 2, 1.0e6)
        end
        labels = [series_label(summary) for summary in panel.series]
        colors = [series_style(summary)[1] for summary in panel.series]
        linestyles = [series_style(summary)[2] for summary in panel.series]
        elements, names = _legend(labels, colors, linestyles; crlb = true)
        Legend(fig[i, 2], elements, names; framevisible = false)
    end
    Label(fig[0, 1:2], title; fontsize = 16)
    save(path, fig)
    return path
end

function save_nees_figure(path, panels, title)
    CairoMakie.activate!(type = "png")
    n = length(panels)
    fig = Figure(size = (1040, 420 * n))
    for (i, panel) in enumerate(panels)
        lower, upper = mean_chi2_interval(panel.dof, panel.M)
        axis = Axis(fig[i, 1]; xlabel = panel.xlabel, ylabel = "average NEES", title = panel.title)
        band!(axis, panel.series[1].times, fill(lower, length(panel.series[1].times)), fill(upper, length(panel.series[1].times)); color = (:gray, 0.28))
        ylims!(axis, 0, max(40, 4 * upper))
        for summary in panel.series
            color, linestyle = series_style(summary)
            lines!(axis, summary.times, summary.nees_series; color = color, linestyle = linestyle, linewidth = 1.8)
        end
        labels = [series_label(summary) for summary in panel.series]
        colors = [series_style(summary)[1] for summary in panel.series]
        linestyles = [series_style(summary)[2] for summary in panel.series]
        elements, names = _legend(labels, colors, linestyles; band = true)
        Legend(fig[i, 2], elements, names; framevisible = false)
    end
    Label(fig[0, 1:2], title; fontsize = 16)
    save(path, fig)
    return path
end

function save_cond_figure(path, panels, title)
    CairoMakie.activate!(type = "png")
    n = length(panels)
    fig = Figure(size = (1040, 420 * n))
    for (i, panel) in enumerate(panels)
        axis = Axis(fig[i, 1]; xlabel = panel.xlabel, ylabel = "median log10 cond(P)", title = panel.title)
        for summary in panel.series
            color, linestyle = series_style(summary)
            y = [isfinite(value) && value > 0 ? log10(value) : NaN for value in summary.cond_series]
            lines!(axis, summary.times, y; color = color, linestyle = linestyle, linewidth = 1.8)
        end
        labels = [series_label(summary) for summary in panel.series]
        colors = [series_style(summary)[1] for summary in panel.series]
        linestyles = [series_style(summary)[2] for summary in panel.series]
        elements, names = _legend(labels, colors, linestyles)
        Legend(fig[i, 2], elements, names; framevisible = false)
    end
    Label(fig[0, 1:2], title; fontsize = 16)
    save(path, fig)
    return path
end

function copy_figure(source, filename)
    destination = joinpath(MEDIA_DIR, filename)
    cp(source, destination; force = true)
    filesize(destination) > 0 || error("$destination is empty")
    return destination
end

function fmt(value)
    return isfinite(value) ? @sprintf("%.6g", value) : "NaN"
end

function write_illcond_table(path, rows)
    open(path, "w") do io
        println(io, "ratio,precision,filter,order,M,n_sigma,position_rmse_km,beta_rmse,nees,nees_in_band,factor_failure_rate,divergence_rate,runtime_per_step_s,mean_log10_cond,rmse_over_pcrlb")
        for row in rows
            println(
                io,
                row.ratio, ",", row.precision, ",", row.name, ",", row.order, ",", row.M, ",",
                row.n_sigma, ",", row.rmse, ",", row.extra_rmse, ",", row.nees, ",",
                row.nees_in_band, ",", row.factor_fail, ",", row.diverge, ",", row.runtime, ",",
                row.mean_log10_cond, ",", row.rmse_over_crlb,
            )
        end
    end
    return path
end

function write_longarc_table(path, rows)
    open(path, "w") do io
        println(io, "variant,requested_gap_h,actual_gap_h,filter,precision,order,M,n_sigma,position_rmse_km,velocity_rmse_km_s,gap_end_position_rmse_km,final_position_rmse_km,nees,nees_in_band,factor_failure_rate,divergence_rate,reacquisition_rate,runtime_per_step_s,mean_log10_cond,rmse_over_pcrlb")
        for row in rows
            println(
                io,
                row.case, ",", row.requested_gap_h, ",", row.actual_gap_h, ",", row.name, ",",
                row.precision, ",", row.order, ",", row.M, ",", row.n_sigma, ",", row.rmse, ",",
                row.extra_rmse, ",", row.gap_rmse, ",", row.final_rmse, ",", row.nees, ",",
                row.nees_in_band, ",", row.factor_fail, ",", row.diverge, ",", row.reacquire, ",",
                row.runtime, ",", row.mean_log10_cond, ",", row.rmse_over_crlb,
            )
        end
    end
    return path
end

function markdown_illcond(rows, M)
    lines = String[]
    push!(lines, "| ratio | precision | filter | n_σ | pos. RMSE (km) | β RMSE | NEES | NEES in band | factor fail | diverge | s/step | mean log10 cond | RMSE/PCRLB |")
    push!(lines, "|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows
        push!(
            lines,
            @sprintf(
                "| %.0e | %s | %s | %d | %s | %s | %s | %s | %.3f | %.3f | %.3e | %s | %s |",
                row.ratio, row.precision, row.name, row.n_sigma, fmt(row.rmse), fmt(row.extra_rmse),
                fmt(row.nees), fmt(row.nees_in_band), row.factor_fail, row.diverge, row.runtime,
                fmt(row.mean_log10_cond), fmt(row.rmse_over_crlb),
            ),
        )
    end
    return join(lines, "\n")
end

function markdown_longarc(rows)
    lines = String[]
    push!(lines, "| variant | gap req. (h) | gap actual (h) | filter | n_σ | pos. RMSE | vel. RMSE | gap-end RMSE | final RMSE | NEES | in band | factor fail | diverge | reacquire | s/step | RMSE/PCRLB |")
    push!(lines, "|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows
        push!(
            lines,
            @sprintf(
                "| %s | %.1f | %.2f | %s | %d | %s | %s | %s | %s | %s | %s | %.3f | %.3f | %.3f | %.3e | %s |",
                row.case, row.requested_gap_h, row.actual_gap_h, row.name, row.n_sigma,
                fmt(row.rmse), fmt(row.extra_rmse), fmt(row.gap_rmse), fmt(row.final_rmse),
                fmt(row.nees), fmt(row.nees_in_band), row.factor_fail, row.diverge, row.reacquire,
                row.runtime, fmt(row.rmse_over_crlb),
            ),
        )
    end
    return join(lines, "\n")
end

function find_row(rows; kwargs...)
    for row in rows
        ok = true
        for (key, value) in pairs(kwargs)
            getfield(row, key) == value || (ok = false)
        end
        ok && return row
    end
    return nothing
end

function illcond_plot_ratio(rows)
    best_ratio = ILL_RATIOS[end]
    best_gap = -Inf
    for ratio in ILL_RATIOS
        for precision in ("Float32", "Float64")
            for order in (4, 6)
                cut = find_row(rows; ratio = ratio, precision = precision, name = "CUT-$order")
                sr = find_row(rows; ratio = ratio, precision = precision, name = "SR-CUT-$order")
                if cut === nothing || sr === nothing
                    continue
                end
                gap = cut.factor_fail - sr.factor_fail
                if gap > best_gap
                    best_gap = gap
                    best_ratio = ratio
                end
            end
        end
    end
    if best_gap <= 0
        for ratio in reverse(ILL_RATIOS)
            sr = find_row(rows; ratio = ratio, precision = "Float64", name = "SR-CUT-4")
            if sr !== nothing && isfinite(sr.rmse)
                return ratio, best_gap
            end
        end
    end
    return best_ratio, best_gap
end

function plot_series(rows, names)
    chosen = Summary[]
    for name in names
        for row in rows
            row.name == name && push!(chosen, row)
        end
    end
    return chosen
end
