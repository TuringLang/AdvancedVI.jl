module AdvancedVIDynamicPPLExt

using ADTypes: ADTypes
using AdvancedVI: AdvancedVI
using AbstractPPL: AbstractPPL
using DynamicPPL: DynamicPPL
using LogDensityProblems: LogDensityProblems
using Random

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

# `LogDensityProblems.capabilities` and the gradient/Hessian methods dispatch
# off `Prep`, so the AD backend's `Prepared` type drives the LDP capability.
struct DynamicPPLModelLogDensityFunction{
    Model<:DynamicPPL.Model,
    VarInfo<:DynamicPPL.AbstractVarInfo,
    ADType<:Union{Nothing,ADTypes.AbstractADType},
    Prep,
}
    model::Model
    varinfo::VarInfo
    adtype::ADType
    model_ref::Ref{Any}
    loglikeadj_ref::Ref{Float64}
    prep::Prep
end

function DynamicPPLModelLogDensityFunction(
    model::DynamicPPL.Model,
    varinfo::DynamicPPL.AbstractVarInfo;
    adtype::Union{Nothing,ADTypes.AbstractADType}=nothing,
    loglikeadj::Real=1.0,
    subsampling::Union{Nothing,AdvancedVI.AbstractSubsampling}=nothing,
)
    model_sub = if isnothing(subsampling)
        model
    else
        rng = Random.default_rng()
        sub_st = AdvancedVI.init(rng, subsampling)
        batch, _, _ = AdvancedVI.step(rng, subsampling, sub_st)
        subsample_dynamicpplmodel(model, batch)
    end

    params = collect(varinfo[:])

    model_ref = Ref{Any}(model_sub)
    loglikeadj_ref = Ref{Float64}(float(loglikeadj))

    prep = if isnothing(adtype)
        nothing
    else
        f = params -> logdensity_impl(params, model_ref[], loglikeadj_ref[], varinfo)
        AbstractPPL.prepare(adtype, f, params)
    end

    return DynamicPPLModelLogDensityFunction(
        model, varinfo, adtype, model_ref, loglikeadj_ref, prep
    )
end

function LogDensityProblems.logdensity(prob::DynamicPPLModelLogDensityFunction, params)
    return logdensity_impl(params, prob.model_ref[], prob.loglikeadj_ref[], prob.varinfo)
end

function LogDensityProblems.logdensity_and_gradient(
    prob::DynamicPPLModelLogDensityFunction, params
)
    return LogDensityProblems.logdensity_and_gradient(prob.prep, params)
end

function LogDensityProblems.logdensity_gradient_and_hessian(
    prob::DynamicPPLModelLogDensityFunction, params
)
    return LogDensityProblems.logdensity_gradient_and_hessian(prob.prep, params)
end

function LogDensityProblems.capabilities(
    ::Type{<:DynamicPPLModelLogDensityFunction{M,V,Nothing,P}}
) where {M,V,P}
    return LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.capabilities(
    ::Type{<:DynamicPPLModelLogDensityFunction{M,V,A,P}}
) where {M,V,A<:ADTypes.AbstractADType,P}
    return LogDensityProblems.capabilities(P)
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

    # Mutates the refs so the previously prepared AD evaluator keeps reading
    # the latest batch without needing a re-prepare.
    prob.model_ref[] = model_sub
    prob.loglikeadj_ref[] = loglikeadj

    return prob
end

end
