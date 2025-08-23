
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
