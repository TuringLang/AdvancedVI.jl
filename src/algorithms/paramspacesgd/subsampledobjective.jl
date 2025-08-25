
"""
    ReshufflingBatchSubsampling(dataset, batchsize)

Random reshuffling subsampling strategy.
At each 'epoch', this strategy splits the dataset into batches of `batchsize`, shuffles the order of the batches, and goes through each of them in that order.
Processing a single batch is referred as a 'step.'
At the end of each epoch, which means we've gone through all the batches exactly once, we repeat the whole process.

# Arguments
- `dataset::AbstractVector`: Iterable sequence representing the dataset.
- `batchsize::Int`: Number of data points in each batch. If the number of data points is not exactly dividable by `batchsize`, the last batch may contain less data points than `batchsize`.
"""
struct ReshufflingBatchSubsampling{DataSet<:AbstractVector} <: AbstractSubsampling
    dataset::DataSet
    batchsize::Int
end

struct ReshufflingBatchSubsamplingState{It}
    epoch::Int
    iterator::It
end

Base.length(sub::ReshufflingBatchSubsampling) = ceil(Int, length(sub.dataset)/sub.batchsize)

function reshuffle_batches(rng::Random.AbstractRNG, sub::ReshufflingBatchSubsampling)
    (; dataset, batchsize) = sub
    shuffled = Random.shuffle(rng, dataset)
    batches = Iterators.partition(shuffled, batchsize)
    return enumerate(batches)
end

function init(rng::Random.AbstractRNG, sub::ReshufflingBatchSubsampling)
    return ReshufflingBatchSubsamplingState(1, reshuffle_batches(rng, sub))
end

function step(
    rng::Random.AbstractRNG,
    sub::ReshufflingBatchSubsampling,
    state::ReshufflingBatchSubsamplingState,
)
    (; epoch, iterator) = state
    (sub_step, batch), batch_it′ = Iterators.peel(iterator)
    epoch′, iterator′′ = if isempty(batch_it′)
        epoch + 1, reshuffle_batches(rng, sub)
    else
        epoch, batch_it′
    end
    info = (epoch=epoch, step=sub_step)
    state′ = ReshufflingBatchSubsamplingState(epoch′, iterator′′)
    return batch, state′, info
end

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
    prob,
    params,
    restructure,
)
    (; objective, subsampling) = subobj
    sub_st = init(rng, subsampling)
    obj_st = AdvancedVI.init(rng, objective, adtype, prob, params, restructure)
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

    batch, sub_st′, sub_inf = step(rng, subsampling, sub_st)
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
