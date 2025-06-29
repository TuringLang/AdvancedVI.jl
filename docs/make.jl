
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
        "Examples" => "examples.md",
        "Parameter Space SGD" => [
            "General" => "paramspacesgd/general.md",
            "Objectives" => [
                "Overview" => "paramspacesgd/objectives.md",
                "RepGradELBO" => "paramspacesgd/repgradelbo.md",
                "ScoreGradELBO" => "paramspacesgd/scoregradelbo.md",
            ],
        ],
        "Variational Families" => "families.md",
        "Optimization" => "optimization.md",
    ],
    warnonly=[:missing_docs],
)
