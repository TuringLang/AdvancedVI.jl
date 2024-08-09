
module AdvancedVIEnzymeExt

if isdefined(Base, :get_extension)
    using Enzyme
    using AdvancedVI
    using AdvancedVI: ADTypes, DiffResults
else
    using ..Enzyme
    using ..AdvancedVI
    using ..AdvancedVI: ADTypes, DiffResults
end

function AdvancedVI.value_and_gradient!(
    ad::ADTypes.AutoEnzyme, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    ∇θ = DiffResults.gradient(out)
    fill!(∇θ, zero(T))
    _, y = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal, f, Enzyme.Active, Enzyme.Duplicated(θ, ∇θ)
    )
    DiffResults.value!(out, y)
    return out
end

end
