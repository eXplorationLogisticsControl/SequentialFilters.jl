"""
Chi-square bands for averaged NEES and NIS.

The average of `M` i.i.d. χ²(n) samples is χ²(M n) / M. The 95% interval uses
the Wilson–Hilferty approximation

```
χ²_p(ν) ≈ ν (1 - 2/(9ν) + z_p sqrt(2/(9ν)))³
```

with `z = ±1.959963984540054`. This stays in the experiment scripts; the core
package does not depend on a special-function library.
"""

const CHI2_Z95 = 1.959963984540054

function chi2_quantile(dof::Real, z::Real)
    ν = Float64(dof)
    ν > 0 || throw(ArgumentError("degrees of freedom must be positive, got $dof"))
    correction = 1 - 2 / (9ν) + Float64(z) * sqrt(2 / (9ν))
    correction > 0 || throw(ArgumentError("Wilson–Hilferty correction is not positive for dof=$dof, z=$z"))
    return ν * correction^3
end

"""95% interval for the mean of `M` χ²(`dof`) samples, as `(lower, upper)`."""
function mean_chi2_interval(dof::Real, M::Integer; z::Real = CHI2_Z95)
    m = Int(M)
    m >= 1 || throw(ArgumentError("M must be positive, got $M"))
    total = Float64(dof) * m
    return chi2_quantile(total, -z) / m, chi2_quantile(total, z) / m
end
