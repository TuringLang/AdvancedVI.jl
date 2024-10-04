using Documenter
using AdvancedVI

# Doctest setup
DocMeta.setdocmeta!(AdvancedVI, :DocTestSetup, :(using AdvancedVI); recursive=true)

makedocs(;
    sitename="AdvancedVI",
    modules=[AdvancedVI],
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
    checkdocs=:exports,
    doctest=false,
)

deploydocs(; repo="github.com/TuringLang/AdvancedVI.jl.git", push_preview=true)
