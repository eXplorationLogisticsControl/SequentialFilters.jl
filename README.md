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

### TODO

- [ ] Higher-order unscented Kalman estimators (HOUSE) variants
- [x] Particle filters (parallelization via `EnsembleProblem`)
- [ ] Gaussian mixture filters
- [ ] Filters with consider parameters
  - [ ] Unscented Schmidt-Kalman filter (USKF), Zanetti and Demars, 2013, Stauch and Jah, 2015.
  - [ ] Square-root unscented Schmidt-Kalman filter (SRUSKF), Geeraert and McMahon 2018.