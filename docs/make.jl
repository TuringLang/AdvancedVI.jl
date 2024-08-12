
using AdvancedVI
using Documenter

DocMeta.setdocmeta!(AdvancedVI, :DocTestSetup, :(using AdvancedVI); recursive=true)

makedocs(;
    modules=[AdvancedVI],
    sitename="AdvancedVI.jl",
    repo="https://github.com/TuringLang/AdvancedVI.jl/blob/{commit}{path}#{line}",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true"),
    pages=[
        "AdvancedVI" => "index.md",
        "General Usage" => "general.md",
        "Examples" => "examples.md",
        "ELBO Maximization" => [
            "Overview" => "elbo/overview.md",
            "Reparameterization Gradient Estimator" => "elbo/repgradelbo.md",
            "Location-Scale Variational Family" => "locscale.md",
        ],
        "Optimization" => "optimization.md",
    ],
)

deploydocs(; repo="github.com/TuringLang/AdvancedVI.jl", push_preview=true)
