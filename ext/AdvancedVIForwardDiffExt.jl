
module AdvancedVIForwardDiffExt

if isdefined(Base, :get_extension)
    using ForwardDiff
    using AdvancedVI
    using AdvancedVI: ADTypes
else
    using ..ForwardDiff
    using ..AdvancedVI
    using ..AdvancedVI: ADTypes
end

getchunksize(::ADTypes.AutoForwardDiff{chunksize}) where {chunksize} = chunksize

function AdvancedVI.value_and_gradient(
    ad::ADTypes.AutoForwardDiff, f, θ::AbstractVector{<:Real}
)
    chunk_size = getchunksize(ad)
    config = if isnothing(chunk_size)
        ForwardDiff.GradientConfig(f, θ)
    else
        ForwardDiff.GradientConfig(f, θ, ForwardDiff.Chunk(length(θ), chunk_size))
    end
    g = ForwardDiff.gradient(f, θ, config)
    g, f(θ)
end

end
