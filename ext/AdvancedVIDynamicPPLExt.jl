module AdvancedVIDynamicPPLExt

using ADTypes: ADTypes
using Accessors
using AdvancedVI: AdvancedVI
using DifferentiationInterface: DifferentiationInterface
using Distributions: Distributions
using DynamicPPL: DynamicPPL
using LogDensityProblems: LogDensityProblems
using Random

adtype_capabilities(::Type{Nothing}) = LogDensityProblems.LogDensityOrder{0}()
function adtype_capabilities(::Type{<:ADTypes.AbstractADType})
    LogDensityProblems.LogDensityOrder{1}()
end

struct DynamicPPLModelLogDensityFunction{
    Model<:DynamicPPL.Model,
    LogLikeAdj<:Real,
    VarInfo<:DynamicPPL.AbstractVarInfo,
    ADType<:ADTypes.AbstractADType,
    PrepGrad<:Union{Nothing,DifferentiationInterface.GradientPrep},
    PrepHess<:Union{Nothing,DifferentiationInterface.HessianPrep},
}
    model::Model
    loglikeadj::LogLikeAdj
    varinfo::VarInfo
    adtype::ADType
    prep_grad::PrepGrad
    prep_hess::PrepHess
end

function logdensity_impl(
    params, model::DynamicPPL.Model, loglikeadj::Real, varinfo::DynamicPPL.AbstractVarInfo
)
    vi = DynamicPPL.unflatten(varinfo, params)
    _, vi = DynamicPPL.evaluate!!(model, vi)
    loglike = DynamicPPL.getloglikelihood(vi)
    logprior = DynamicPPL.getlogprior(vi)
    logjac = DynamicPPL.getlogjac(vi)
    return convert(eltype(params), loglikeadj)*loglike + logprior - logjac
end

function subsample_dynamicpplmodel(model::DynamicPPL.Model, batch)
    return DynamicPPL.decondition(
        model, DynamicPPL.@varname(datapoints)) | (; datapoints=batch)
end

function DynamicPPLModelLogDensityFunction(
    model::DynamicPPL.Model,
    varinfo::DynamicPPL.AbstractVarInfo;
    adtype::Union{Nothing,ADTypes.AbstractADType}=nothing,
    loglikeadj::Real=1.0,
    subsampling::Union{Nothing,AdvancedVI.AbstractSubsampling}=nothing,
)
    if !DynamicPPL.is_supported(adtype)
        @warn "The AD backend $adtype is not officially supported by DynamicPPL. Gradient calculations may still work, but compatibility is not guaranteed."
    end
    
    model_sub = if isnothing(subsampling)
        model
    else
        rng = Random.default_rng()
        sub_st = AdvancedVI.init(rng, subsampling)
        batch, _, _ = AdvancedVI.step(rng, subsampling, sub_st)
        subsample_dynamicpplmodel(model, batch)
    end

    params = [val for val in varinfo[:]]
    prep_grad =
        if adtype_capabilities(typeof(adtype)) >= LogDensityProblems.LogDensityOrder{1}()
            DifferentiationInterface.prepare_gradient(
                logdensity_impl,
                adtype,
                params,
                DifferentiationInterface.Constant(model_sub),
                DifferentiationInterface.Constant(loglikeadj),
                DifferentiationInterface.Constant(varinfo),
            )
        else
            nothing
        end
    prep_hess =
        if adtype_capabilities(typeof(adtype)) > LogDensityProblems.LogDensityOrder{2}()
            DifferentiationInterface.prepare_hessian(
                logdensity_impl,
                adtype,
                params,
                DifferentiationInterface.Constant(model_sub),
                DifferentiationInterface.Constant(loglikeadj),
                DifferentiationInterface.Constant(varinfo),
            )
        else
            nothing
        end
    return DynamicPPLModelLogDensityFunction{
        typeof(model),
        typeof(loglikeadj),
        typeof(varinfo),
        typeof(adtype),
        typeof(prep_grad),
        typeof(prep_hess),
    }(
        model, loglikeadj, varinfo, adtype, prep_grad, prep_hess
    )
end

function LogDensityProblems.logdensity(prob::DynamicPPLModelLogDensityFunction, params)
    (; model, loglikeadj, varinfo) = prob
    return logdensity_impl(params, model, loglikeadj, varinfo)
end

function LogDensityProblems.logdensity_and_gradient(
    prob::DynamicPPLModelLogDensityFunction, params
)
    (; model, adtype, loglikeadj, varinfo, prep_grad) = prob
    return DifferentiationInterface.value_and_gradient(
        logdensity_impl,
        prep_grad,
        adtype,
        params,
        DifferentiationInterface.Constant(model),
        DifferentiationInterface.Constant(loglikeadj),
        DifferentiationInterface.Constant(varinfo),
    )
end

function LogDensityProblems.logdensity_gradient_and_hessian(
    prob::DynamicPPLModelLogDensityFunction, params
)
    (; model, adtype, loglikeadj, varinfo, prep_hess) = prob
    return DifferentiationInterface.value_gradient_and_hessian(
        logdensity_impl,
        prep_hess,
        adtype,
        params,
        DifferentiationInterface.Constant(model),
        DifferentiationInterface.Constant(loglikeadj),
        DifferentiationInterface.Constant(varinfo),
    )
end

function LogDensityProblems.capabilities(
    ::Type{<:DynamicPPLModelLogDensityFunction{M,L,V,ADType,PG,PH}}
) where {M,L,V,ADType<:ADTypes.AbstractADType,PG,PH}
    return adtype_capabilities(ADType)
end

function LogDensityProblems.dimension(prob::DynamicPPLModelLogDensityFunction)
    return length(prob.varinfo[:])
end

function AdvancedVI.subsample(prob::DynamicPPLModelLogDensityFunction, batch)
    model = prob.model

    if !haskey(model.defaults, :datapoints)
        throw(
            ArgumentError(
                "Subsampling is turned on, but the model does not have have a `datapoints` keyword argument.",
            ),
        )
    end

    n_datapoints = length(model.defaults.datapoints)
    batchsize = length(batch)
    model_sub = subsample_dynamicpplmodel(model, batch)
    loglikeadj = n_datapoints/batchsize

    prob′ = @set prob.model = model_sub
    prob′′ = @set prob′.loglikeadj = loglikeadj
    return prob′′
end

end
