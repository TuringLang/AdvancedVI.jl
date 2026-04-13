module AdvancedVIForwardDiffExt

using ADTypes: ADTypes
using AdvancedVI: AdvancedVI
using ForwardDiff

const DiffResults = ForwardDiff.DiffResults

function AdvancedVI._prepare_gradient(
    f, adtype::ADTypes.AutoForwardDiff{chunk_size}, x::AbstractVector{<:Real}, aux
) where {chunk_size}
    f = Base.Fix2(f, aux)
    chunk = if chunk_size === nothing
        ForwardDiff.Chunk(length(x))
    else
        ForwardDiff.Chunk(length(x), chunk_size)
    end
    cfg = if isnothing(adtype.tag)
        ForwardDiff.GradientConfig(f, x, chunk)
    else
        ForwardDiff.GradientConfig(f, x, chunk, adtype.tag)
    end
    result = DiffResults.GradientResult(similar(x))
    return (; cfg, result)
end

function AdvancedVI._value_and_gradient(
    f, prep, ::ADTypes.AutoForwardDiff, x::AbstractVector{<:Real}, aux
)
    f = Base.Fix2(f, aux)
    ForwardDiff.gradient!(prep.result, f, x, prep.cfg, Val(false))
    return DiffResults.value(prep.result), copy(DiffResults.gradient(prep.result))
end

end
