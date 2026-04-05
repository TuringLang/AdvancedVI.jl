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
    return LogDensityProblems.LogDensityOrder{1}()
end

function adtype_capabilities(
    ::Type{
        <:Union{
            <:ADTypes.AutoForwardDiff,
            <:ADTypes.AutoReverseDiff,
            <:ADTypes.AutoMooncake,
            <:ADTypes.AutoEnzyme,
            <:DifferentiationInterface.SecondOrder,
        },
    },
)
    return LogDensityProblems.LogDensityOrder{2}()
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

function DynamicPPLModelLogDensityFunction(
    model::DynamicPPL.Model,
    varinfo::DynamicPPL.AbstractVarInfo;
    use_hessian::Bool=true,
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

    params = [val for val in varinfo[:]]
    cap = adtype_capabilities(typeof(adtype))
    prep_grad = if cap >= LogDensityProblems.LogDensityOrder{1}()
        DifferentiationInterface.prepare_gradient(
            logdensity_impl,
            DifferentiationInterface.inner(adtype),
            params,
            DifferentiationInterface.Constant(model_sub),
            DifferentiationInterface.Constant(loglikeadj),
            DifferentiationInterface.Constant(varinfo),
        )
    else
        nothing
    end
    prep_hess = if cap >= LogDensityProblems.LogDensityOrder{2}() && use_hessian
        try
            DifferentiationInterface.prepare_hessian(
                logdensity_impl,
                adtype,
                params,
                DifferentiationInterface.Constant(model_sub),
                DifferentiationInterface.Constant(loglikeadj),
                DifferentiationInterface.Constant(varinfo),
            )
        catch
            @warn "The selected AD backend has second-order capabilities but `DifferentiationInterface.prepare_hessian` failed. AdvancedVI will treat the model to only have first-order capability."
            nothing
        end
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
        DifferentiationInterface.inner(adtype),
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
    return if PH != Nothing
        LogDensityProblems.LogDensityOrder{2}()
    elseif PG != Nothing
        LogDensityProblems.LogDensityOrder{1}()
    else
        LogDensityProblems.LogDensityOrder{0}()
    end
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
    loglikeadj = n_datapoints / batchsize

    prob′ = @set prob.model = model_sub
    prob′′ = @set prob′.loglikeadj = loglikeadj
    return prob′′
end

end
