
function AdvancedVI.grad!(
    f::Function,
    ::AutoEnzyme,
    λ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    )
    # Use `Enzyme.ReverseWithPrimal` once it is released:
    # https://github.com/EnzymeAD/Enzyme.jl/pull/598
    y = f(λ)
    DiffResults.value!(out, y)
    dy = DiffResults.gradient(out)
    fill!(dy, 0)
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, f, Enzyme.Active, Enzyme.Duplicated(λ, dy))
    return out
end
