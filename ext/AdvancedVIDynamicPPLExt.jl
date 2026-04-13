module AdvancedVIDynamicPPLExt

using ADTypes: ADTypes
using Accessors
using AdvancedVI: AdvancedVI
using Distributions: Distributions
using DynamicPPL: DynamicPPL
using LogDensityProblems: LogDensityProblems
using Random

adtype_capabilities(::Type{Nothing}) = LogDensityProblems.LogDensityOrder{0}()

function adtype_capabilities(::Type{<:ADTypes.AbstractADType})
    return LogDensityProblems.LogDensityOrder{1}()
end

struct DynamicPPLModelLogDensityFunction{
    Model<:DynamicPPL.Model,
    LogLikeAdj<:Real,
    VarInfo<:DynamicPPL.AbstractVarInfo,
    ADType,
    GradADType,
    PrepGrad,
    PrepHess,
}
    model::Model
    loglikeadj::LogLikeAdj
    varinfo::VarInfo
    adtype::ADType
    grad_adtype::GradADType
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

function logdensity_impl(params, aux::NamedTuple{(:model, :loglikeadj, :varinfo)})
    return logdensity_impl(params, aux.model, aux.loglikeadj, aux.varinfo)
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
    grad_adtype = adtype isa ADTypes.AutoMooncake ? ADTypes.AutoForwardDiff() : adtype
    cap = adtype_capabilities(typeof(grad_adtype))
    aux = (model=model_sub, loglikeadj=loglikeadj, varinfo=varinfo)
    di_ext = Base.get_extension(AdvancedVI, :AdvancedVIDifferentiationInterfaceExt)
    prep_grad = if !isnothing(di_ext) && adtype isa di_ext.DI.SecondOrder
        di_ext.DI.prepare_gradient(
            logdensity_impl, di_ext.DI.inner(adtype), params, di_ext.DI.Constant(aux)
        )
    elseif cap >= LogDensityProblems.LogDensityOrder{1}()
        AdvancedVI._prepare_gradient(logdensity_impl, grad_adtype, params, aux)
    else
        nothing
    end
    prep_hess = if !isnothing(di_ext) && adtype isa di_ext.DI.SecondOrder && use_hessian
        try
            di_ext.DI.prepare_hessian(logdensity_impl, adtype, params, di_ext.DI.Constant(aux))
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
        typeof(grad_adtype),
        typeof(prep_grad),
        typeof(prep_hess),
    }(
        model, loglikeadj, varinfo, adtype, grad_adtype, prep_grad, prep_hess
    )
end

function LogDensityProblems.logdensity(prob::DynamicPPLModelLogDensityFunction, params)
    (; model, loglikeadj, varinfo) = prob
    return logdensity_impl(params, model, loglikeadj, varinfo)
end

function LogDensityProblems.logdensity_and_gradient(
    prob::DynamicPPLModelLogDensityFunction, params
)
    (; model, adtype, grad_adtype, loglikeadj, varinfo, prep_grad) = prob
    aux = (model=model, loglikeadj=loglikeadj, varinfo=varinfo)
    di_ext = Base.get_extension(AdvancedVI, :AdvancedVIDifferentiationInterfaceExt)
    if !isnothing(di_ext) && adtype isa di_ext.DI.SecondOrder
        return di_ext.DI.value_and_gradient(
            logdensity_impl,
            prep_grad,
            di_ext.DI.inner(adtype),
            params,
            di_ext.DI.Constant(aux),
        )
    end
    return AdvancedVI._value_and_gradient(
        logdensity_impl, prep_grad, grad_adtype, params, aux
    )
end

function LogDensityProblems.logdensity_gradient_and_hessian(
    prob::DynamicPPLModelLogDensityFunction, params
)
    (; model, adtype, loglikeadj, varinfo, prep_hess) = prob
    isnothing(prep_hess) && throw(
        MethodError(LogDensityProblems.logdensity_gradient_and_hessian, (prob, params))
    )
    di_ext = Base.get_extension(AdvancedVI, :AdvancedVIDifferentiationInterfaceExt)
    isnothing(di_ext) && throw(
        ArgumentError(
            "Load `DifferentiationInterface` to use second-order DynamicPPL backends."
        ),
    )
    aux = (model=model, loglikeadj=loglikeadj, varinfo=varinfo)
    return di_ext.DI.value_gradient_and_hessian(
        logdensity_impl, prep_hess, adtype, params, di_ext.DI.Constant(aux)
    )
end

function LogDensityProblems.capabilities(
    ::Type{<:DynamicPPLModelLogDensityFunction{M,L,V,ADType,GADType,PG,PH}}
) where {M,L,V,ADType,GADType,PG,PH}
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
