
# This function/signature will be moved to src/AdvancedVI.jl
"""
    subsample(model, batch)

# Arguments
- `model`: Model subject to subsampling. Could be the target model or the variational approximation.
- `batch`: Data points or indices corresponding to the subsampled "batch."

# Returns 
- `sub`: Subsampled model.
"""
subsample(model::Any, ::Any) = model

struct Subsampling{
    O<:AbstractVariationalObjective,D<:AbstractVector
} <: AbstractVariationalObjective
    batchsize::Int
    objective::O
    data::D
end

function init_batch(rng::Random.AbstractRNG, data::AbstractVector, batchsize::Int)
    shuffled = Random.shuffle(rng, data)
    batches = Iterators.partition(shuffled, batchsize)
    return enumerate(batches)
end

function AdvancedVI.init(rng::Random.AbstractRNG, sub::Subsampling, params, restructure)
    @unpack batchsize, objective, indices = sub
    epoch = 1
    sub_state = (epoch, init_batch(rng, indices, batchsize))
    obj_state = AdvancedVI.init(rng, objective, params, restructure)
    return (sub_state, obj_state)
end

function next_batch(rng::Random.AbstractRNG, sub::Subsampling, sub_state)
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

function estimate_gradient!(
    rng::Random.AbstractRNG,
    sub::Subsampling,
    adtype::ADTypes.AbstractADType,
    out::DiffResults.MutableDiffResult,
    prob,
    params,
    restructure,
    state,
)
    obj = sub.objective
    sub_st, obj_st = state
    q = restructure(params)

    batch, sub_st′, sub_stat = next_batch(rng, sub, sub_st)
    prob_sub = subsample(prob, batch)
    q_sub = subsample(q, batch)
    params_sub, re_sub = Optimisers.destructure(q_sub)

    out, obj_st′, obj_stat = AdvancedVI.estimate_gradient!(
        rng, obj, adtype, out, prob_sub, params, params_sub, re_sub, obj_st
    )
    return out, (sub_st′, obj_st′), merge(sub_stat, obj_stat)
end
