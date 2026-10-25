"""
Posterior Cramér–Rao bound of Tichavský, Muravchik, and Nehorai (1998)
for additive Gaussian noise.

The expectation is replaced by the noise-free reference trajectory and the
scenario's truth Jacobians (the matched model, including when the filter's
μ is misspecified):

```
J₀ = P₀⁻¹
P⁻ = Q + F J⁻¹ F'
J  ← (P⁻)⁻¹ + H' R⁻¹ H
```

`F` is the state transition over the step that ends at the measurement.
`H` is the measurement Jacobian at that measurement. Both matrices are
symmetrized. The bound on a block RMSE is `sqrt(tr(P[block]))`.
"""

function posterior_information(scenario::Scenario)
    nx = scenario.nx
    n_steps = scenario.n_steps
    covariances = Array{Float64}(undef, nx, nx, n_steps + 1)
    covariances[:, :, 1] = (scenario.P0 + scenario.P0') ./ 2
    J = Symmetric((covariances[:, :, 1]) \ I(nx))
    for k in 1:n_steps
        Fk = scenario.F_truth(scenario.times[k], scenario.reference_truth[:, k], scenario.Δt)
        predicted = scenario.Q + Fk * (J \ Fk')
        predicted = Symmetric((predicted + predicted') ./ 2)
        Hk = scenario.H_truth(scenario.times[k + 1], scenario.reference_truth[:, k + 1])
        information = Matrix(predicted \ I(nx)) + Matrix(Hk)' * Matrix(scenario.R \ Hk)
        information = (information + information') ./ 2
        J = Symmetric(information)
        posterior = Matrix(J \ I(nx))
        covariances[:, :, k + 1] = (posterior + posterior') ./ 2
    end
    return covariances
end

"""`sqrt(tr(P[indices]))` at each time, including the prior."""
function crlb_rmse(covariances::AbstractArray{<:Real,3}, indices)
    idx = indices isa Integer ? (indices:indices) : indices
    bound = Vector{Float64}(undef, size(covariances, 3))
    for k in 1:size(covariances, 3)
        block = covariances[idx, idx, k]
        bound[k] = sqrt(max(tr(block), 0.0))
    end
    return bound
end
