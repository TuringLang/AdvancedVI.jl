
module AdvancedVIEnzymeExt

if isdefined(Base, :get_extension)
    using Enzyme
    using AdvancedVI
    using AdvancedVI: ADTypes
else
    using ..Enzyme
    using ..AdvancedVI
    using ..AdvancedVI: ADTypes
end

# Enzyme doesn't support f::Bijectors (see https://github.com/EnzymeAD/Enzyme.jl/issues/916)
function AdvancedVI.value_and_gradient(
    ad::ADTypes.AutoEnzyme, f, θ::AbstractVector{T}
) where {T<:Real}
    y = f(θ)
    ∇θ = similar(θ)
    fill!(∇θ, zero(T))
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, f, Enzyme.Active, Enzyme.Duplicated(θ, ∇θ))
    ∇θ,  y
end

end
