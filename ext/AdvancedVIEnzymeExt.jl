
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

function AdvancedVI.stop_gradient(::ADTypes.AutoEnzyme, x)
    throw("Score function estimator with Enzyme is not supported yet.")
end

# Enzyme doesn't support f::Bijectors (see https://github.com/EnzymeAD/Enzyme.jl/issues/916)
function AdvancedVI.value_and_gradient!(
    ad::ADTypes.AutoEnzyme, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    y = f(θ)
    DiffResults.value!(out, y)
    ∇θ = DiffResults.gradient(out)
    fill!(∇θ, zero(T))
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, f, Enzyme.Active, Enzyme.Duplicated(θ, ∇θ))
    return out
end

end
