
using AdvancedVI
using Documenter

DocMeta.setdocmeta!(
    AdvancedVI, :DocTestSetup, :(using AdvancedVI); recursive=true
)

makedocs(;
    modules  = [AdvancedVI],
    sitename = "AdvancedVI.jl",
    repo     = "https://github.com/TuringLang/AdvancedVI.jl/blob/{commit}{path}#{line}",
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    pages    = ["AdvancedVI"        => "index.md",
                "Getting Started"   => "started.md",
                "ELBO Maximization" => [
                    "Black-Box Variational Inference" => "bbvi.md",   
                    "Location Scale Family"           => "locscale.md",
                ]],
)

deploydocs(; repo="github.com/TuringLang/AdvancedVI.jl", push_preview=true)
