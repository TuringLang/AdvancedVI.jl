
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
    ad ::ADTypes.AutoForwardDiff,
    f,
    x  ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult
)
    chunk_size = getchunksize(ad)
    config = if isnothing(chunk_size)
        ForwardDiff.GradientConfig(f, x)
    else
        ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk(length(x), chunk_size))
    end
    ForwardDiff.gradient!(out, f, x, config)
    return out
end

function AdvancedVI.value_and_gradient!(
    ad ::ADTypes.AutoForwardDiff,
    f,
    x  ::AbstractVector,
    aux, 
    out::DiffResults.MutableDiffResult
)
    AdvancedVI.value_and_gradient!(ad, x′ -> f(x′, aux), x, out)
end

end
