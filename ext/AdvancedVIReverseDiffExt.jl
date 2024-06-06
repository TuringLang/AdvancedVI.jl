
module AdvancedVIReverseDiffExt

if isdefined(Base, :get_extension)
    using AdvancedVI
    using AdvancedVI: ADTypes
    using ReverseDiff
else
    using ..AdvancedVI
    using ..AdvancedVI: ADTypes
    using ..ReverseDiff
end

# ReverseDiff without compiled tape
function AdvancedVI.value_and_gradient(
    ad::ADTypes.AutoReverseDiff, f, θ::AbstractVector{<:Real}
)
    tp = ReverseDiff.GradientTape(f, θ)
    g = ReverseDiff.gradient!(tp, θ)
    g, f(θ)
end

end
