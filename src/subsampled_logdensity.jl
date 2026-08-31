"""
    SubsampledLogDensity(prob, make_prob, dataset_size)

Wrap a `LogDensityProblem` so `subsample(prob, batch)` replaces its inner
problem with `make_prob(batch, dataset_size / length(batch))`. The inner
problem's capabilities and dimension are forwarded. `make_prob` must preserve
the concrete problem type and dimension, and runs once per batch.

`dataset_size` must be positive and equal the population size from which
`batch` is drawn. `batch` must be nonempty.
"""
struct SubsampledLogDensity{P,F}
    prob::P
    make_prob::F
    dataset_size::Int
    function SubsampledLogDensity{P,F}(prob::P, make_prob::F, dataset_size::Int) where {P,F}
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

function LogDensityProblems.logdensity(wrapper::SubsampledLogDensity, x)
    return LogDensityProblems.logdensity(wrapper.prob, x)
end

function LogDensityProblems.logdensity_and_gradient(wrapper::SubsampledLogDensity, x)
    return LogDensityProblems.logdensity_and_gradient(wrapper.prob, x)
end

function LogDensityProblems.logdensity_gradient_and_hessian(
    wrapper::SubsampledLogDensity, x
)
    return LogDensityProblems.logdensity_gradient_and_hessian(wrapper.prob, x)
end

function LogDensityProblems.dimension(wrapper::SubsampledLogDensity)
    return LogDensityProblems.dimension(wrapper.prob)
end

function LogDensityProblems.capabilities(::Type{<:SubsampledLogDensity{P}}) where {P}
    return LogDensityProblems.capabilities(P)
end

function subsample(wrapper::SubsampledLogDensity{P,F}, batch) where {P,F}
    batch_size = length(batch)
    batch_size > 0 || throw(ArgumentError("`batch` must be nonempty."))
    dim = LogDensityProblems.dimension(wrapper.prob)
    new_inner = wrapper.make_prob(batch, wrapper.dataset_size / batch_size)
    new_dim = LogDensityProblems.dimension(new_inner)
    new_dim == dim || throw(
        DimensionMismatch(
            "`make_prob` changed the log-density dimension from $dim to $new_dim; " *
            "the dimension must remain invariant across batches.",
        ),
    )
    return SubsampledLogDensity{P,F}(new_inner, wrapper.make_prob, wrapper.dataset_size)
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
