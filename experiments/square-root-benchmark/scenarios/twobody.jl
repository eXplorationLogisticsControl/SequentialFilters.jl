"""
Two-body range and range-rate scenario from the EKF demo.

Orbit, sensor, and noise match `demo/twobody/demo_ekf.jl`:

  μ = 3.986004418e5 km³/s²,  R_E = 6378.137 km,
  ω_E = 7.292115146706979e-5 rad/s,  a = 7000 km,  inc = 51.6°,
  station latitude 28.5°, longitude 0 at t = 0.

The sample interval is one tenth of the period (`steps_per_orbit = 10`) and
the arc is two orbits (`n_meas = 20`). Measurement standard deviations are
`σ_ρ = 0.01` km and `σ_ρ̇ = 1e-5` km/s. The initial covariance is diagonal
with `σ_r = 1` km and `σ_v = 1e-3` km/s. Process noise is
`UnbiasedRandomProcess(1e-6)`.

Each Monte Carlo draw adds a random initial error and discrete process noise
`w ~ N(0, Q)` after the deterministic two-body step, then measurement noise.
The posterior Cramér–Rao bound uses the noise-free reference. A run diverges
when the position error exceeds `divergence_threshold` kilometres (default
50) or the estimate or covariance is not finite.

`measurement_noise_scale` multiplies the measurement-noise standard deviation
and the `R` given to the filter. `initial_error_scale` does the same for the
initial error and `P0`. `model_mismatch` multiplies only the filter's μ.
`steps_per_orbit` and `n_orbits` change the sample interval and the arc length.
"""

const TWOBODY_MU = 3.986004418e5
const TWOBODY_EARTH_RADIUS = 6378.137
const TWOBODY_EARTH_RATE = 7.292115146706979e-5
const TWOBODY_SMA = 7000.0
const TWOBODY_INCLINATION = deg2rad(51.6)
const TWOBODY_LATITUDE = deg2rad(28.5)
const TWOBODY_LONGITUDE0 = 0.0

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

function twobody_initial_state()
    speed = sqrt(TWOBODY_MU / TWOBODY_SMA)
    return [
        TWOBODY_SMA,
        0.0,
        0.0,
        0.0,
        speed * cos(TWOBODY_INCLINATION),
        speed * sin(TWOBODY_INCLINATION),
    ]
end

function twobody_station(t)
    cφ = cos(TWOBODY_LATITUDE)
    sφ = sin(TWOBODY_LATITUDE)
    λ = TWOBODY_LONGITUDE0 + TWOBODY_EARTH_RATE * t
    cλ = cos(λ)
    sλ = sin(λ)
    rs = TWOBODY_EARTH_RADIUS .* (cφ * cλ, cφ * sλ, sφ)
    vs = (TWOBODY_EARTH_RATE * TWOBODY_EARTH_RADIUS) .* (-cφ * sλ, cφ * cλ, 0.0)
    return rs, vs
end

function twobody_range_rate(t, x)
    rs, vs = twobody_station(t)
    ρx = x[1] - rs[1]
    ρy = x[2] - rs[2]
    ρz = x[3] - rs[3]
    ρ = sqrt(ρx * ρx + ρy * ρy + ρz * ρz)
    ρdot = (ρx * (x[4] - vs[1]) + ρy * (x[5] - vs[2]) + ρz * (x[6] - vs[3])) / ρ
    return [ρ, ρdot]
end

function twobody_H(t, x)
    return ForwardDiff.jacobian(z -> twobody_range_rate(t, z), collect(Float64, x))
end

function twobody_h_H(t, x)
    return twobody_range_rate(t, x), twobody_H(t, x)
end

const TWOBODY_DYNAMICS = SequentialFilters.NonlinearODEModel(twobody!; n = 6, p = TWOBODY_MU)
const TWOBODY_PROCESS_NOISE = SequentialFilters.UnbiasedRandomProcess(1.0e-6)
const TWOBODY_MEASUREMENTS = SequentialFilters.NonlinearMeasurementModel(twobody_h_H)

function twobody_scenario(;
    measurement_noise_scale::Real = 1.0,
    initial_error_scale::Real = 1.0,
    steps_per_orbit::Integer = 10,
    n_orbits::Integer = 2,
    model_mismatch::Real = 1.0,
    divergence_threshold::Real = 50.0,
)
    scale_y = Float64(measurement_noise_scale)
    scale_x = Float64(initial_error_scale)
    mismatch = Float64(model_mismatch)
    scale_y > 0 || throw(ArgumentError("measurement_noise_scale must be positive"))
    scale_x > 0 || throw(ArgumentError("initial_error_scale must be positive"))
    mismatch > 0 || throw(ArgumentError("model_mismatch must be positive"))
    steps = Int(steps_per_orbit)
    orbits = Int(n_orbits)
    steps >= 1 || throw(ArgumentError("steps_per_orbit must be positive"))
    orbits >= 1 || throw(ArgumentError("n_orbits must be positive"))

    period = 2π * sqrt(TWOBODY_SMA^3 / TWOBODY_MU)
    Δt = period / steps
    n_meas = steps * orbits
    x0 = twobody_initial_state()
    nx = length(x0)

    σ_ρ = 0.01
    σ_ρdot = 1.0e-5
    R = Matrix(scale_y^2 * Diagonal([σ_ρ^2, σ_ρdot^2]))
    σ_r = 1.0
    σ_v = 1.0e-3
    P0_nominal = Matrix(Diagonal([fill(σ_r^2, 3); fill(σ_v^2, 3)]))
    P0 = scale_x^2 * P0_nominal

    μ_filter = TWOBODY_MU * mismatch
    if mismatch == 1.0
        dynamics = TWOBODY_DYNAMICS
        dynamics_truth = TWOBODY_DYNAMICS
    else
        dynamics = SequentialFilters.NonlinearODEModel(twobody!; n = nx, p = μ_filter)
        dynamics_truth = TWOBODY_DYNAMICS
    end
    process_noise = TWOBODY_PROCESS_NOISE
    Q = Matrix(SequentialFilters.get_noise(process_noise, Δt))

    f = StateStep(twobody!, μ_filter, nx)
    f_variational, F = variational_pair(dynamics, process_noise)
    f_truth = StateStep(twobody!, TWOBODY_MU, nx)
    _, F_truth = variational_pair(dynamics_truth, process_noise)

    reference = noise_free_trajectory(f_truth, x0, n_meas, Δt)
    times = collect(range(0.0, step = Δt, length = n_meas + 1))
    L0 = Matrix(cholesky(Hermitian(P0_nominal)).L)
    Lq = Matrix(cholesky(Hermitian((Q + Q') ./ 2)).L)
    Lr = Matrix(cholesky(Hermitian(R)).L)

    function draw(rng)
        xhat0 = x0 .+ scale_x .* (L0 * randn(rng, nx))
        truth = Matrix{Float64}(undef, nx, n_meas + 1)
        truth[:, 1] = x0
        x = copy(x0)
        t = 0.0
        ys = Vector{Vector{Float64}}(undef, n_meas)
        for k in 1:n_meas
            x = f_truth(t, x, Δt) .+ Lq * randn(rng, nx)
            t += Δt
            truth[:, k + 1] = x
            ys[k] = twobody_range_rate(t, x) .+ Lr * randn(rng, 2)
        end
        return (; truth, ys, xhat0, P0 = copy(P0))
    end

    return Scenario(
        "two-body",
        nx,
        2,
        n_meas,
        Δt,
        (position = 1:3, velocity = 4:6),
        Float64(divergence_threshold),
        1:3,
        P0,
        Q,
        R,
        f,
        f_variational,
        F,
        twobody_range_rate,
        twobody_H,
        dynamics,
        process_noise,
        TWOBODY_MEASUREMENTS,
        draw,
        reference,
        f_truth,
        F_truth,
        twobody_H,
        times,
        (;
            measurement_noise_scale = scale_y,
            initial_error_scale = scale_x,
            steps_per_orbit = steps,
            n_orbits = orbits,
            model_mismatch = mismatch,
            divergence_threshold = Float64(divergence_threshold),
        ),
    )
end
