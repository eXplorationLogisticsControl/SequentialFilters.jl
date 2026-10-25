"""
Lorenz-63 case shared by the filter demos.

The vector field is the standard chaotic Lorenz system, not a Lorentz force:

    dx/dt = σ (y − x),  dy/dt = x (ρ − z) − y,  dz/dt = x y − β z,

with σ = 10, ρ = 28, and β = 8/3. `lorenz_dynamics()` is a `NonlinearODEModel`
accepted by `predict!`.

The initial condition `x0 = [1, 1, 1]` is off the attractor. The nonzero
equilibria sit near `(±√(β(ρ−1)), ±√(β(ρ−1)), ρ−1) ≈ (±8.49, ±8.49, 27)`.
The origin is a saddle, so this nearby state leaves the origin and falls
onto the butterfly.

The filter step is `Δt = 0.02` for 2000 steps (final time 40). Process noise
is additive, `Q = 10⁻³ I`. The measurement is the partial state `(x, z)` with
independent unit-variance noise. The measurement seed is `Random.seed!(1)`.
The filter mean starts at `x0 + [5, −5, 8]`.
"""

using LinearAlgebra
using OrdinaryDiffEq
using Random

using SequentialFilters

const LORENZ_Σ = 10.0
const LORENZ_Ρ = 28.0
const LORENZ_Β = 8.0 / 3.0
const LORENZ_PARAMETERS = (LORENZ_Σ, LORENZ_Ρ, LORENZ_Β)

"""In-place Lorenz-63 vector field. `p` is `(σ, ρ, β)`."""
function lorenz63!(du, u, p, t)
    σ, ρ, β = p
    x, y, z = u
    du[1] = σ * (y - x)
    du[2] = x * (ρ - z) - y
    du[3] = x * y - β * z
    return nothing
end

"""Dynamics model the Lorenz demos pass to `predict!`."""
function lorenz_dynamics()
    return NonlinearODEModel(
        lorenz63!;
        n = 3,
        p = LORENZ_PARAMETERS,
        reltol = 1e-5,
        abstol = 1e-5,
    )
end

"""
    load_lorenz_case(; n_particles=500)

Truth, measurements, noise, and the shared dynamics model.

`n_particles` is only the particle-filter demo's count. The trajectory and
the measurements do not depend on it.
"""
function load_lorenz_case(; n_particles::Integer = 500)
    Δt = 0.02
    n_steps = 2000
    x0 = [1.0, 1.0, 1.0]
    t_final = n_steps * Δt
    truth = solve(
        ODEProblem(lorenz63!, x0, (0.0, t_final), LORENZ_PARAMETERS),
        Tsit5();
        reltol = 1e-10,
        abstol = 1e-10,
        saveat = Δt,
    )

    σ_meas = 1.0
    R = Diagonal(fill(σ_meas^2, 2))
    Random.seed!(1)
    ys = map(1:n_steps) do k
        state = truth(k * Δt)
        return [state[1], state[3]] .+ σ_meas .* randn(2)
    end

    x_hat0 = x0 .+ [5.0, -5.0, 8.0]
    P0 = Matrix(Diagonal(fill(25.0, 3)))
    dynamics = lorenz_dynamics()
    process_noise = AdditiveProcessNoise(Matrix(1.0e-3 * I(3)))
    h(t, x) = [x[1], x[3]]
    function get_h_H(t, x)
        H = zeros(2, 3)
        H[1, 1] = 1.0
        H[2, 3] = 1.0
        return h(t, x), H
    end
    measurements = NonlinearMeasurementModel(get_h_H)

    return (;
        Δt, n_steps, x0, x_hat0, P0, R, ys, truth, h, dynamics, process_noise,
        measurements, n_particles, σ_meas,
    )
end

"""Print posterior root-mean-square error and the final posterior error."""
function report_lorenz_errors(post_errors)
    E = reduce(hcat, post_errors)
    npost = size(E, 2)
    rms = [sqrt(sum(abs2, @view(E[i, :])) / npost) for i in 1:3]
    final = E[:, end]
    println("posterior rms (x, y, z) = $(rms[1]) $(rms[2]) $(rms[3])")
    println("final error (x, y, z) = $(final[1]) $(final[2]) $(final[3])")
    return rms, final
end
