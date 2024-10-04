using Documenter
using AdvancedVI

# Doctest setup
DocMeta.setdocmeta!(AdvancedVI, :DocTestSetup, :(using AdvancedVI); recursive=true)

makedocs(;
    sitename="AdvancedVI",
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
    doctest=false,
)

deploydocs(; repo="github.com/TuringLang/AdvancedVI.jl.git", push_preview=true)
