"""
    SubsampledLogDensity(prob, make_prob, dataset_size)

`LogDensityProblems`-compatible wrapper that supports [`with_batch`](@ref):
`with_batch(prob, batch)` returns a fresh wrapper whose inner problem is
`make_prob(batch, dataset_size / length(batch))`. `make_prob` must return
objects of the same concrete type as the initial `prob` so the wrapper stays
type-stable. The inner problem's capabilities and dimension are surfaced.

`dataset_size` must equal the size of the dataset that `batch` indexes into:
the rescaling `dataset_size / length(batch)` is only an unbiased estimator
when these are consistent. `with_batch` checks `length(batch) <= dataset_size`
to catch the obvious misuse; a `batch` drawn from a different dataset will
silently scale the gradient by the wrong factor.
"""
struct SubsampledLogDensity{P,F}
    prob::P
    make_prob::F
    dataset_size::Int
    function SubsampledLogDensity{P,F}(prob::P, make_prob::F, dataset_size::Int) where {P,F}
        # Caught here to prevent silent zero-gradient (or sign flip) downstream.
        dataset_size > 0 ||
            throw(ArgumentError("`dataset_size` must be positive, got $dataset_size."))
        return new{P,F}(prob, make_prob, dataset_size)
    end
end
function SubsampledLogDensity(prob, make_prob, dataset_size::Integer)
    return SubsampledLogDensity{typeof(prob),typeof(make_prob)}(
        prob, make_prob, Int(dataset_size)
    )
end

function LogDensityProblems.logdensity(prob::SubsampledLogDensity, x)
    return LogDensityProblems.logdensity(prob.prob, x)
end

function LogDensityProblems.logdensity_and_gradient(prob::SubsampledLogDensity, x)
    return LogDensityProblems.logdensity_and_gradient(prob.prob, x)
end

function LogDensityProblems.dimension(prob::SubsampledLogDensity)
    return LogDensityProblems.dimension(prob.prob)
end

function LogDensityProblems.capabilities(::Type{<:SubsampledLogDensity{P}}) where {P}
    return LogDensityProblems.capabilities(P)
end

function with_batch(prob::SubsampledLogDensity{P,F}, batch) where {P,F}
    length(batch) <= prob.dataset_size || throw(
        ArgumentError(
            "`length(batch) = $(length(batch))` exceeds `dataset_size = $(prob.dataset_size)`; " *
            "the batch must come from the same dataset that `dataset_size` describes.",
        ),
    )
    new_inner = prob.make_prob(batch, prob.dataset_size / length(batch))
    return SubsampledLogDensity{P,F}(new_inner, prob.make_prob, prob.dataset_size)
end

"""
    WeightedLogJoint(scale)

Callable returning `scale * loglikelihood + logprior - logjacobian` of a
varinfo. The call method is backend-specific; package extensions register
overloads for the varinfo types they support.
"""
struct WeightedLogJoint{T<:Real}
    scale::T
end
