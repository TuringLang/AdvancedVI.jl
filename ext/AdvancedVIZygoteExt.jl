
module AdvancedVIZygoteExt

if isdefined(Base, :get_extension)
    using AdvancedVI
    using AdvancedVI: ADTypes, DiffResults
    using Zygote
else
    using ..AdvancedVI
    using ..AdvancedVI: ADTypes, DiffResults
    using ..Zygote
end

function AdvancedVI.value_and_gradient!(
    ad::ADTypes.AutoZygote, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    y, back = Zygote.pullback(f, θ)
    ∇θ = back(one(T))
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, first(∇θ))
    return out
end

end
