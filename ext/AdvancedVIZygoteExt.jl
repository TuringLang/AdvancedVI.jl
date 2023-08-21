
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
    ad::ADTypes.AutoZygote, f, θ::AbstractVector{<:Real}, out::DiffResults.MutableDiffResult
)
    y, back = Zygote.pullback(f, θ)
    ∇θ = back(one(y))
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, only(∇θ))
    return out
end

end
