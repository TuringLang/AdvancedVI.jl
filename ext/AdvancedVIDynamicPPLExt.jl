module AdvancedVIDynamicPPLExt

using ADTypes: ADTypes
using Accessors
using AdvancedVI: AdvancedVI
using AbstractPPL: AbstractPPL
using Distributions: Distributions
using DynamicPPL: DynamicPPL
using LogDensityProblems: LogDensityProblems
using Random

function adtype_capabilities(::Type{Nothing})
    return LogDensityProblems.LogDensityOrder{0}()
end

function adtype_capabilities(::Type{<:ADTypes.AbstractADType})
    return LogDensityProblems.LogDensityOrder{1}()
end

function logdensity_impl(
    params, model::DynamicPPL.Model, loglikeadj::Real, varinfo::DynamicPPL.AbstractVarInfo
)
    vi = DynamicPPL.unflatten!!(varinfo, params)
    _, vi = DynamicPPL.evaluate!!(model, vi)
    loglike = DynamicPPL.getloglikelihood(vi)
    logprior = DynamicPPL.getlogprior(vi)
    logjac = DynamicPPL.getlogjac(vi)
    return convert(eltype(params), loglikeadj) * loglike + logprior - logjac
end

function subsample_dynamicpplmodel(
    model::DynamicPPL.Model{F,A,D,M,Ta,Td,Ctx,Threaded}, batch
) where {F,A,D,M,Ta,Td,Ctx,Threaded}
    new_kwargs = merge(model.defaults, (; datapoints=batch))
    return DynamicPPL.Model{Threaded}(model.f, model.args, new_kwargs, model.context)
end

struct DynamicPPLModelLogDensityFunction{
    Model<:DynamicPPL.Model,
    VarInfo<:DynamicPPL.AbstractVarInfo,
    ADType<:Union{Nothing,ADTypes.AbstractADType},
    PrepGrad,
}
    model::Model
    varinfo::VarInfo
    adtype::ADType
    # Refs are updated in-place by subsample; the prepared AD evaluator reads
    # through them on every call, so the prep remains valid across subsampling.
    model_ref::Ref{Any}
    loglikeadj_ref::Ref{Float64}
    prep_grad::PrepGrad
end

function DynamicPPLModelLogDensityFunction(
    model::DynamicPPL.Model,
    varinfo::DynamicPPL.AbstractVarInfo;
    use_hessian::Bool=false,
    adtype::Union{Nothing,ADTypes.AbstractADType}=nothing,
    loglikeadj::Real=1.0,
    subsampling::Union{Nothing,AdvancedVI.AbstractSubsampling}=nothing,
)
    use_hessian && @warn "`use_hessian` is no longer supported and will be ignored."
    model_sub = if isnothing(subsampling)
        model
    else
        rng = Random.default_rng()
        sub_st = AdvancedVI.init(rng, subsampling)
        batch, _, _ = AdvancedVI.step(rng, subsampling, sub_st)
        subsample_dynamicpplmodel(model, batch)
    end

    params = [val for val in varinfo[:]]
    cap = adtype_capabilities(typeof(adtype))

    model_ref = Ref{Any}(model_sub)
    loglikeadj_ref = Ref{Float64}(float(loglikeadj))

    prep_grad = if cap >= LogDensityProblems.LogDensityOrder{1}()
        AbstractPPL.prepare(
            adtype,
            params -> logdensity_impl(params, model_ref[], loglikeadj_ref[], varinfo),
            params,
        )
    else
        nothing
    end

    return DynamicPPLModelLogDensityFunction(
        model, varinfo, adtype, model_ref, loglikeadj_ref, prep_grad
    )
end

function LogDensityProblems.logdensity(prob::DynamicPPLModelLogDensityFunction, params)
    return logdensity_impl(params, prob.model_ref[], prob.loglikeadj_ref[], prob.varinfo)
end

function LogDensityProblems.logdensity_and_gradient(
    prob::DynamicPPLModelLogDensityFunction, params
)
    return AbstractPPL.value_and_gradient(prob.prep_grad, params)
end

function LogDensityProblems.capabilities(
    ::Type{<:DynamicPPLModelLogDensityFunction{M,V,Nothing,G}}
) where {M,V,G}
    return LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.capabilities(
    ::Type{<:DynamicPPLModelLogDensityFunction{M,V,<:ADTypes.AbstractADType,G}}
) where {M,V,G}
    return LogDensityProblems.LogDensityOrder{1}()
end

function LogDensityProblems.dimension(prob::DynamicPPLModelLogDensityFunction)
    return length(prob.varinfo[:])
end

function AdvancedVI.subsample(prob::DynamicPPLModelLogDensityFunction, batch)
    model = prob.model

    if !haskey(model.defaults, :datapoints)
        throw(
            ArgumentError(
                "Subsampling is turned on, but the model does not have a `datapoints` keyword argument.",
            ),
        )
    end

    n_datapoints = length(model.defaults.datapoints)
    batchsize = length(batch)
    model_sub = subsample_dynamicpplmodel(model, batch)
    loglikeadj = n_datapoints / batchsize

    prob.model_ref[] = model_sub
    prob.loglikeadj_ref[] = loglikeadj

    return prob
end

end
