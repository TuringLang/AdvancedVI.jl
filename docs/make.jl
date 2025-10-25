
using AdvancedVI
using Documenter

# Necessary for invoking the docstring specializations
using Random
using ADTypes

DocMeta.setdocmeta!(AdvancedVI, :DocTestSetup, :(using AdvancedVI); recursive=true)

makedocs(;
    modules=[AdvancedVI],
    sitename="AdvancedVI.jl",
    repo="https://github.com/TuringLang/AdvancedVI.jl/blob/{commit}{path}#{line}",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true"),
    pages=[
        "AdvancedVI" => "index.md",
        "General Usage" => "general.md",
        "Tutorials" => [
            "Basic Example" => "tutorials/basic.md",
            "Scaling to Large Datasets" => "tutorials/subsampling.md",
            "Stan Models" => "tutorials/stan.md",
            "Normalizing Flows" => "tutorials/flows.md",
        ],
        "Algorithms" => [
            "`KLMinRepGradDescent`" => "klminrepgraddescent.md",
            "`KLMinRepGradProxDescent`" => "klminrepgradproxdescent.md",
            "`KLMinScoreGradDescent`" => "klminscoregraddescent.md",
            "`KLMinWassFwdBwd`" => "klminwassfwdbwd.md",
        ],
        "Variational Families" => "families.md",
        "Optimization" => "optimization.md",
    ],
    warnonly=[:missing_docs],
)
