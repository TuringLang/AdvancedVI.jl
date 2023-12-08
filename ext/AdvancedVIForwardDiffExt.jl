
module AdvancedVIForwardDiffExt

if isdefined(Base, :get_extension)
    using ForwardDiff
    using AdvancedVI
    using AdvancedVI: ADTypes, DiffResults
else
    using ..ForwardDiff
    using ..AdvancedVI
    using ..AdvancedVI: ADTypes, DiffResults
end

getchunksize(::ADTypes.AutoForwardDiff{chunksize}) where {chunksize} = chunksize

function AdvancedVI.value_and_gradient!(
    ad::ADTypes.AutoForwardDiff, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    chunk_size = getchunksize(ad)
    config = if isnothing(chunk_size)
        ForwardDiff.GradientConfig(f, θ)
    else
        ForwardDiff.GradientConfig(f, θ, ForwardDiff.Chunk(length(θ), chunk_size))
    end
    ForwardDiff.gradient!(out, f, θ, config)
    return out
end

end
