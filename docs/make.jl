#using AdvancedVI
using Documenter

DocMeta.setdocmeta!(
    AdvancedVI, :DocTestSetup, :(using AdvancedVI); recursive=true
)

makedocs(;
    sitename = "AdvancedVI.jl",
    modules  = [AdvancedVI],
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
         pages    = ["index.md",
                     "families.md",
                     "advi.md"],
)

deploydocs(; repo="github.com/TuringLang/AdvancedVI.jl", devbranch="main")
