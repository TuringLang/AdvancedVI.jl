
"""
    Subsampled(objective, batchsize, data)

Subsample `objective` over the dataset represented by `data` with minibatches of size `batchsize`.

# Arguments
- `objective::AbstractVariationalObjective`: A variational objective that is compatible with subsampling.
- `batchsize::Int`: Size of minibatches.
- `data`: An iterator over the datapoints or indices representing the datapoints.
"""
struct Subsampled{O<:AbstractVariationalObjective,D<:AbstractVector} <:
       AbstractVariationalObjective
    objective::O
    batchsize::Int
    data::D
end

function init_batch(rng::Random.AbstractRNG, data::AbstractVector, batchsize::Int)
    shuffled = Random.shuffle(rng, data)
    batches = Iterators.partition(shuffled, batchsize)
    return enumerate(batches)
end

function AdvancedVI.init(
    rng::Random.AbstractRNG, sub::Subsampled, prob, params, restructure
)
    @unpack batchsize, objective, data = sub
    epoch = 1
    sub_state = (epoch, init_batch(rng, data, batchsize))
    obj_state = AdvancedVI.init(rng, objective, prob, params, restructure)
    return (sub_state, obj_state)
end

function next_batch(rng::Random.AbstractRNG, sub::Subsampled, sub_state)
    epoch, batch_itr = sub_state
    (step, batch), batch_itr′ = Iterators.peel(batch_itr)
    epoch′, batch_itr′′ = if isempty(batch_itr′)
        epoch + 1, init_batch(rng, sub.data, sub.batchsize)
    else
        epoch, batch_itr′
    end
    stat = (epoch=epoch, step=step)
    return batch, (epoch′, batch_itr′′), stat
end

function estimate_objective(
    rng::Random.AbstractRNG,
    sub::Subsampled,
    q,
    prob;
    n_batches::Int=ceil(Int, length(sub.data) / sub.batchsize),
    kwargs...,
)
    @unpack objective, batchsize, data = sub
    sub_st = (1, init_batch(rng, data, batchsize))
    return mean(1:n_batches) do _
        batch, sub_st, _ = next_batch(rng, sub, sub_st)
        prob_sub = subsample(prob, batch)
        q_sub = subsample(q, batch)
        estimate_objective(rng, objective, q_sub, prob_sub; kwargs...)
    end
end

function estimate_objective(
    sub::Subsampled,
    q,
    prob;
    n_batches::Int=ceil(Int, length(sub.data) / sub.batchsize),
    kwargs...,
)
    return estimate_objective(Random.default_rng(), sub, q, prob; n_batches, kwargs...)
end

function estimate_gradient!(
    rng::Random.AbstractRNG,
    sub::Subsampled,
    adtype::ADTypes.AbstractADType,
    out::DiffResults.MutableDiffResult,
    prob,
    params,
    restructure,
    state,
    objargs...;
    kwargs...,
)
    obj = sub.objective
    sub_st, obj_st = state
    q = restructure(params)

    batch, sub_st′, sub_stat = next_batch(rng, sub, sub_st)
    prob_sub = subsample(prob, batch)
    q_sub = subsample(q, batch)
    params_sub, re_sub = Optimisers.destructure(q_sub)

    out, obj_st′, obj_stat = AdvancedVI.estimate_gradient!(
        rng, obj, adtype, out, prob_sub, params_sub, re_sub, obj_st, objargs...; kwargs...
    )
    return out, (sub_st′, obj_st′), merge(sub_stat, obj_stat)
end
