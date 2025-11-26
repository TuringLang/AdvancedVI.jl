module AdvancedVIDynamicPPLExt

using Accessors
using Distributions: Distributions
using DynamicPPL: DynamicPPL
using AdvancedVI: AdvancedVI


struct AdjustedLogLikelihoodAccumulator{T} <: DynamicPPL.LogProbAccumulator{T}
    "the scalar log likelihood value"
    logp::T

    "adjustment factor to be multplied to the cummulative log-likelihood"
    adj::T
end

DynamicPPL.logp(acc::AdjustedLogLikelihoodAccumulator) = acc.logp

DynamicPPL.accumulator_name(::Type{<:AdjustedLogLikelihoodAccumulator}) = :LogLikelihood

DynamicPPL.accumulate_assume!!(acc::AdjustedLogLikelihoodAccumulator, val, logjac, vn, right) = acc

function DynamicPPL.accumulate_observe!!(acc::AdjustedLogLikelihoodAccumulator, right, left, vn)
    # Note that it's important to use the loglikelihood function here, not logpdf, because
    # they handle vectors differently:
    # https://github.com/JuliaStats/Distributions.jl/issues/1972
    return DynamicPPL.acclogp(acc, Distributions.loglikelihood(right, left))
end

function DynamicPPL.convert_eltype(::Type{T}, acc::AdjustedLogLikelihoodAccumulator) where {T}
    return AdjustedLogLikelihoodAccumulator(convert(T, DynamicPPL.logp(acc)), convert(T, acc.adj))
end

function DynamicPPL._zero(acc::AdjustedLogLikelihoodAccumulator{T}) where {T}
    return AdjustedLogLikelihoodAccumulator(zero(T), acc.adj)
end

function DynamicPPL.acclogp(acc::AdjustedLogLikelihoodAccumulator, val)
    return AdjustedLogLikelihoodAccumulator(DynamicPPL.logp(acc) + val, acc.adj)
end


function AdvancedVI.subsample(prob::DynamicPPL.LogDensityFunction, batch_idx)
    (; model, getlogdensity, varinfo, adtype) = prob

    @assert haskey(model.defaults, :datapoints) "Subsampling is turned on, but the model does not have have a `datapoints` keyword argument."

    n_datapoints = length(model.defaults.datapoints)
    batch = model.defaults.datapoints[batch_idx]
    model_subsampled = @set model.defaults.datapoints = batch
    batchsize = length(batch_idx)

    logprior_acc, logjac_acc, _ = DynamicPPL.getaccs(varinfo)

    T = typeof(DynamicPPL.logp(logprior_acc))
    adj = convert(T, n_datapoints/batchsize)
    loglikacc_adj = AdjustedLogLikelihoodAccumulator(zero(T), adj)

    accs′ = (logprior_acc, logjac_acc, loglikacc_adj)
    varinfo′ = DynamicPPL.setaccs!!(varinfo, accs′)

    # As of DynamicPPL 0.38.9, the constructor below calls to `DI.prepare`.
    # DynamicPPL#1156 is expected to relax this so that we can simply
    # mutate the updated files with Accessors.@set
    prob′′ = DynamicPPL.LogDensityFunction(
        model_subsampled, getlogdensity, varinfo′; adtype=adtype
    )
    return prob′′
end

end
