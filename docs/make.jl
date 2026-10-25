using Documenter
using SequentialFilters

DocMeta.setdocmeta!(
    SequentialFilters,
    :DocTestSetup,
    :(using LinearAlgebra, SequentialFilters);
    recursive = true,
)

makedocs(;
    modules = [SequentialFilters],
    authors = "Yuri Shimane",
    sitename = "SequentialFilters.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://explorationlogisticscontrol.github.io/SequentialFilters.jl",
        edit_link = "main",
    ),
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Filters" => [
            "Extended Kalman filter (EKF)" => "filters/ekf.md",
            "Unscented Kalman filter (UKF)" => "filters/ukf.md",
            "Conjugate unscented filter (CUT)" => "filters/cut.md",
            "Square-root EKF (SREKF)" => "filters/srekf.md",
            "Square-root UKF (SRUKF)" => "filters/srukf.md",
            "Square-root CUT (SRCUT)" => "filters/srcut.md",
            "Square-root CKF (SRCKF)" => "filters/srckf.md",
            "Bootstrap particle filter (PF)" => "filters/pf.md",
        ],
        "API" => "api.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(;
    repo = "github.com/eXplorationLogisticsControl/SequentialFilters.jl.git",
    devbranch = "main",
    push_preview = true,
)
