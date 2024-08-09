
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
    ad::ADTypes.AutoReverseDiff,
    f,
    x::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
)
    tp = ReverseDiff.GradientTape(f, x)
    ReverseDiff.gradient!(out, tp, x)
    return out
end

function AdvancedVI.value_and_gradient!(
    ad::ADTypes.AutoReverseDiff,
    f,
    x::AbstractVector{<:Real},
    aux,
    out::DiffResults.MutableDiffResult,
)
    return AdvancedVI.value_and_gradient!(ad, x′ -> f(x′, aux), x, out)
end

end
