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

function Base.length(sub::ReshufflingBatchSubsampling)
    return ceil(Int, length(sub.dataset) / sub.batchsize)
end

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
    drop_trailing_batch_if_too_small::Bool=false,
)
    (; epoch, iterator) = state
    (sub_step, batch), iterator = Iterators.peel(iterator)
    if isempty(iterator)
        iterator = reshuffle_batches(rng, sub)
        if drop_trailing_batch_if_too_small && length(batch) < sub.batchsize
            # Ignore the trailing batch if its size is smaller than `batchsize`.
            # This should only be used when estimating gradients during optimization.
            # This is necessary to ensure that all batches have the same size.
            # Otherwise, `DifferentiationInterface.prepare_*` behaves incorrectly.
            (sub_step, batch), iterator = Iterators.peel(iterator)
        end
        epoch = epoch + 1
    end
    info = (epoch=epoch, step=sub_step)
    state = ReshufflingBatchSubsamplingState(epoch, iterator)
    return batch, state, info
end
