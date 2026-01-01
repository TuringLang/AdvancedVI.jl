"""
    SubsampledObjective(objective, subsampling)

Subsample `objective` according to the `subsampling` strategy.

# Arguments
- `objective::AbstractVariationalObjective`: A variational objective that is compatible with subsampling.
- `subsampling::AbstractSubsampling`: Subsampling strategy.
"""
struct SubsampledObjective{Obj<:AbstractVariationalObjective,Sub<:AbstractSubsampling} <:
       AbstractVariationalObjective
    objective::Obj
    subsampling::Sub
end

struct SubsampledObjectiveState{Prob,SubSt,ObjSt}
    prob::Prob
    sub_st::SubSt
    obj_st::ObjSt
end

function init(
    rng::Random.AbstractRNG,
    subobj::SubsampledObjective,
    adtype::ADTypes.AbstractADType,
    q_init,
    prob,
    params,
    restructure,
)
    (; objective, subsampling) = subobj
    sub_st = init(rng, subsampling)

    # This is necessary to ensure that `init` sees the type "conditioned" on a minibatch
    # when calling `DifferentiationInterface.prepare_*` inside it.
    batch, _, _ = step(rng, subsampling, sub_st, true)
    prob_sub = subsample(prob, batch)
    q_init_sub = subsample(q_init, batch)
    params_sub, re_sub = Optimisers.destructure(q_init_sub)

    obj_st = AdvancedVI.init(
        rng, objective, adtype, q_init_sub, prob_sub, params_sub, re_sub
    )
    return SubsampledObjectiveState(prob, sub_st, obj_st)
end

function estimate_objective(
    rng::Random.AbstractRNG, subobj::SubsampledObjective, q, prob; kwargs...
)
    (; objective, subsampling) = subobj
    sub_st = init(rng, subsampling)
    return mapreduce(+, 1:length(subsampling)) do _
        batch, sub_st, _ = step(rng, subsampling, sub_st)
        prob_sub = subsample(prob, batch)
        q_sub = subsample(q, batch)
        estimate_objective(rng, objective, q_sub, prob_sub; kwargs...) / length(subsampling)
    end
end

function estimate_objective(subobj::SubsampledObjective, q, prob; kwargs...)
    return estimate_objective(Random.default_rng(), subobj, q, prob; kwargs...)
end

function estimate_gradient!(
    rng::Random.AbstractRNG,
    subobj::SubsampledObjective,
    adtype::ADTypes.AbstractADType,
    out::DiffResults.MutableDiffResult,
    state::SubsampledObjectiveState,
    params,
    restructure,
    args...;
    kwargs...,
)
    (; objective, subsampling) = subobj
    (; prob, sub_st, obj_st) = state
    q = restructure(params)

    batch, sub_st′, sub_inf = step(rng, subsampling, sub_st, true)
    prob_sub = subsample(prob, batch)
    q_sub = subsample(q, batch)
    params_sub, re_sub = Optimisers.destructure(q_sub)

    obj_st′ = set_objective_state_problem(obj_st, prob_sub)
    out, obj_st′′, obj_inf = AdvancedVI.estimate_gradient!(
        rng, objective, adtype, out, obj_st′, params_sub, re_sub, args...; kwargs...
    )
    state′ = SubsampledObjectiveState(prob, sub_st′, obj_st′′)
    return out, state′, merge(sub_inf, obj_inf)
end
