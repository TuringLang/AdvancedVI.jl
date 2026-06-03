module AdvancedVIDynamicPPLExt

using ADTypes: ADTypes
using AdvancedVI: AdvancedVI
using AbstractPPL: AbstractPPL
using DynamicPPL: DynamicPPL
using LogDensityProblems: LogDensityProblems
using Random

adtype_capabilities(::Type{Nothing}) = LogDensityProblems.LogDensityOrder{0}()

function adtype_capabilities(::Type{<:ADTypes.AbstractADType})
    return LogDensityProblems.LogDensityOrder{1}()
end

# `getlogdensity` callable for `DynamicPPL.logdensity_internal`: reads the
# current `loglikeadj` through a Ref so the mutation done by `subsample` is
# observed without rebuilding any AD prep.
struct WeightedLogJoint{R<:Base.RefValue{<:Real}}
    loglikeadj_ref::R
end
function (g::WeightedLogJoint)(vi)
    loglike = DynamicPPL.getloglikelihood(vi)
    logprior = DynamicPPL.getlogprior(vi)
    logjac = DynamicPPL.getlogjac(vi)
    return g.loglikeadj_ref[] * loglike + logprior - logjac
end

const _DEFAULT_LDF_ACCS = DynamicPPL.AccumulatorTuple((
    DynamicPPL.LogPriorAccumulator(),
    DynamicPPL.LogJacobianAccumulator(),
    DynamicPPL.LogLikelihoodAccumulator(),
))

function subsample_dynamicpplmodel(
    model::DynamicPPL.Model{F,A,D,M,Ta,Td,Ctx,Threaded}, batch
) where {F,A,D,M,Ta,Td,Ctx,Threaded}
    new_kwargs = merge(model.defaults, (; datapoints=batch))
    return DynamicPPL.Model{Threaded}(model.f, model.args, new_kwargs, model.context)
end

# `model` is the original (unsubsampled) source of truth; `subsample` must read
# it (not `model_ref[]`) to get the full-dataset length on every call.
# `model_ref`/`loglikeadj_ref` are mutated in place by `subsample` so the
# closure inside `prep_grad`/`prep_hess` stays valid across subsampling steps.
# `model_ref` is `Ref{Any}` because `subsample_dynamicpplmodel`'s output type
# varies with the batch (a typed Ref would throw on reassignment), and because
# compiled-tape backends would otherwise bake the deref into the tape and miss
# the `subsample` update.
struct DynamicPPLModelLogDensityFunction{
    Model<:DynamicPPL.Model,
    LogLikeAdj<:Real,
    Ranges<:DynamicPPL.VarNamedTuple,
    Strategy<:DynamicPPL.AbstractTransformStrategy,
    GetLogDensity,
    ADType<:Union{Nothing,ADTypes.AbstractADType},
    PrepGrad,
    PrepHess,
}
    model::Model
    model_ref::Ref{Any}
    loglikeadj_ref::Ref{LogLikeAdj}
    ranges_and_transforms::Ranges
    transform_strategy::Strategy
    getlogdensity::GetLogDensity
    adtype::ADType
    prep_grad::PrepGrad
    prep_hess::PrepHess
    dim::Int
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

    ranges_and_transforms, params = DynamicPPL.get_rat_and_samplevec(varinfo.values)
    transform_strategy = DynamicPPL.infer_transform_strategy_from_values(
        ranges_and_transforms
    )

    model_ref = Ref{Any}(model_sub)
    loglikeadj_ref = Ref(float(loglikeadj))
    getlogdensity = WeightedLogJoint(loglikeadj_ref)
    f =
        params -> DynamicPPL.logdensity_internal(
            params,
            model_ref[],
            getlogdensity,
            ranges_and_transforms,
            transform_strategy,
            _DEFAULT_LDF_ACCS,
        )
    cap = adtype_capabilities(typeof(adtype))

    prep_grad = if cap >= LogDensityProblems.LogDensityOrder{1}()
        AbstractPPL.prepare(adtype, f, params)
    else
        nothing
    end
    prep_hess = if cap >= LogDensityProblems.LogDensityOrder{1}() && use_hessian
        try
            AbstractPPL.prepare(adtype, f, params; order=2)
        catch err
            err isa MethodError || rethrow()
            @warn "The selected AD backend does not support `AbstractPPL.prepare(...; order=2)`. AdvancedVI will treat the model as first-order only."
            nothing
        end
    else
        nothing
    end
    return DynamicPPLModelLogDensityFunction{
        typeof(model),
        typeof(loglikeadj_ref[]),
        typeof(ranges_and_transforms),
        typeof(transform_strategy),
        typeof(getlogdensity),
        typeof(adtype),
        typeof(prep_grad),
        typeof(prep_hess),
    }(
        model,
        model_ref,
        loglikeadj_ref,
        ranges_and_transforms,
        transform_strategy,
        getlogdensity,
        adtype,
        prep_grad,
        prep_hess,
        length(params),
    )
end

function LogDensityProblems.logdensity(prob::DynamicPPLModelLogDensityFunction, params)
    return DynamicPPL.logdensity_internal(
        params,
        prob.model_ref[],
        prob.getlogdensity,
        prob.ranges_and_transforms,
        prob.transform_strategy,
        _DEFAULT_LDF_ACCS,
    )
end

# `!!` may alias internal buffers of `prep_*`; copy so callers can retain the
# arrays past the next AD call.
function LogDensityProblems.logdensity_and_gradient(
    prob::DynamicPPLModelLogDensityFunction, params
)
    val, grad = AbstractPPL.value_and_gradient!!(prob.prep_grad, params)
    return val, copy(grad)
end

function LogDensityProblems.logdensity_gradient_and_hessian(
    prob::DynamicPPLModelLogDensityFunction, params
)
    val, grad, H = AbstractPPL.value_gradient_and_hessian!!(prob.prep_hess, params)
    return val, copy(grad), copy(H)
end

function LogDensityProblems.capabilities(
    ::Type{<:DynamicPPLModelLogDensityFunction{Model,L,R,S,G,ADType,PG,PH}}
) where {Model,L,R,S,G,ADType<:Union{Nothing,ADTypes.AbstractADType},PG,PH}
    return if PH !== Nothing
        LogDensityProblems.LogDensityOrder{2}()
    elseif PG !== Nothing
        LogDensityProblems.LogDensityOrder{1}()
    else
        LogDensityProblems.LogDensityOrder{0}()
    end
end

LogDensityProblems.dimension(prob::DynamicPPLModelLogDensityFunction) = prob.dim

function AdvancedVI.subsample(prob::DynamicPPLModelLogDensityFunction, batch)
    model = prob.model  # full dataset — `model_ref[]` would already be subsampled

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
    T = eltype(prob.loglikeadj_ref)
    loglikeadj = T(n_datapoints) / T(batchsize)

    prob.model_ref[] = model_sub
    prob.loglikeadj_ref[] = loglikeadj

    return prob
end

end
