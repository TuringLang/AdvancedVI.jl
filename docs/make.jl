
using AdvancedVI
using Documenter

DocMeta.setdocmeta!(
    AdvancedVI, :DocTestSetup, :(using AdvancedVI); recursive=true
)

makedocs(;
    sitename = "AdvancedVI.jl",
    modules  = [AdvancedVI],
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
         pages    = ["AdvancedVI"        => "index.md",
                     "Getting Started"   => "started.md",
                     "ELBO Maximization" => [
                         "Automatic Differentiation VI" => "advi.md",   
                         "Location Scale Family"        => "locscale.md",
                     ]],
)

deploydocs(; repo="github.com/TuringLang/AdvancedVI.jl", push_preview=true)
