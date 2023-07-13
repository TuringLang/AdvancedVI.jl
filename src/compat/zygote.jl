
function AdvancedVI.grad!(
    f::Function,
    ::AutoZygote,
    λ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    )
    y, back = Zygote.pullback(f, λ)
    dy = first(back(1.0))
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, dy)
    return out
end
