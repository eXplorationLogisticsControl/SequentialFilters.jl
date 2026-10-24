# `SequentialFilters.jl`: Sequential filters in Julia

<p align="center">
  <a href="https://explorationlogisticscontrol.github.io/SequentialFilters.jl">
    <img src="https://github.com/eXplorationLogisticsControl/SequentialFilters.jl/actions/workflows/docs.yml/badge.svg" alt="docs workflow"/>
  </a>
  <a href="https://github.com/eXplorationLogisticsControl/SequentialFilters.jl/actions/workflows/test.yml">
    <img src="https://github.com/eXplorationLogisticsControl/SequentialFilters.jl/actions/workflows/test.yml/badge.svg" alt="test workflow"/>
  </a>
  <!-- <a href="https://app.codecov.io/gh/eXplorationLogisticsControl/SequentialFilters.jl">
    <img src="https://codecov.io/gh/eXplorationLogisticsControl/SequentialFilters.jl/branch/main/graph/badge.svg" alt="codecov"/>
  </a> -->
</p>

Sequential filters for nonlinear state estimation: an extended Kalman filter, a scaled unscented Kalman filter, and square-root forms of both.

Tutorials for each filter are in the [documentation](https://explorationlogisticscontrol.github.io/SequentialFilters.jl). Build it locally with

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The two-body demos print elapsed time, process CPU time, and allocations for each covariance-form filter and its square-root form:

```bash
julia --project=demo demo/demo_twobody_ekf.jl
julia --project=demo demo/demo_ukf.jl
julia --project=demo demo/demo_srekf.jl
julia --project=demo demo/demo_srukf.jl
julia --project=demo demo/demo_srcut.jl
```