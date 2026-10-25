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
        "Tutorials" => [
            "Extended Kalman filter" => "tutorials/ekf.md",
            "Unscented Kalman filter" => "tutorials/ukf.md",
            "Square-root EKF" => "tutorials/srekf.md",
            "Square-root UKF" => "tutorials/srukf.md",
            "Square-root CUT" => "tutorials/srcut.md",
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
