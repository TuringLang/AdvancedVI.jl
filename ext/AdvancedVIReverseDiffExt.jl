
module AdvancedVIReverseDiffExt

if isdefined(Base, :get_extension)
    using AdvancedVI
    using AdvancedVI: ADTypes, DiffResults
    using ReverseDiff
else
    using ..AdvancedVI
    using ..AdvancedVI: ADTypes, DiffResults
    using ..ReverseDiff
end

# ReverseDiff without compiled tape
function AdvancedVI.value_and_gradient!(
    ad::ADTypes.AutoReverseDiff, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    tp = ReverseDiff.GradientTape(f, θ)
    ReverseDiff.gradient!(out, tp, θ)
    return out
end

end
