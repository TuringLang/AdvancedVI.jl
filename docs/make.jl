
using AdvancedVI
using Documenter

DocMeta.setdocmeta!(
    AdvancedVI, :DocTestSetup, :(using AdvancedVI); recursive=true
)

makedocs(;
    sitename = "AdvancedVI.jl",
    modules  = [AdvancedVI],
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
         pages    = ["Home"     => "index.md",
                     "Families" => "families.md",
                     "ADVI"     => "advi.md"],
)

deploydocs(; repo="github.com/TuringLang/AdvancedVI.jl", push_preview=true)
