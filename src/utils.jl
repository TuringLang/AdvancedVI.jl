
function pm_next!(pm, stats::NamedTuple)
    ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end

function maybe_init_optimizer(
    state_init::NamedTuple,
    optimizer ::Optimisers.AbstractRule,
    λ         ::AbstractVector
)
    haskey(state_init, :optimizer) ? state_init.optimizer : Optimisers.setup(optimizer, λ)
end

function maybe_init_objective(
    state_init::NamedTuple,
    rng       ::Random.AbstractRNG,
    objective ::AbstractVariationalObjective,
    λ         ::AbstractVector,
    restructure
)
    haskey(state_init, :objective) ? state_init.objective : init(rng, objective, λ, restructure)
end

eachsample(samples::AbstractMatrix) = eachcol(samples)

eachsample(samples::AbstractVector) = samples

function catsamples_and_acc(
    state_curr::Tuple{<:AbstractArray,  <:Real},
    state_new ::Tuple{<:AbstractVector, <:Real}
)
    x  = hcat(first(state_curr), first(state_new))
    ∑y = last(state_curr) + last(state_new)
    return (x, ∑y)
end

