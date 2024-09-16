
function pm_next!(pm, stats::NamedTuple)
    return ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end

function maybe_init_optimizer(
    state_init::NamedTuple, optimizer::Optimisers.AbstractRule, params
)
    if haskey(state_init, :optimizer)
        state_init.optimizer
    else
        Optimisers.setup(optimizer, params)
    end
end

function maybe_init_averager(state_init::NamedTuple, averager::AbstractAverager, params)
    if haskey(state_init, :averager)
        state_init.averager
    else
        init(averager, params)
    end
end

function maybe_init_objective(
    state_init::NamedTuple,
    rng::Random.AbstractRNG,
    objective::AbstractVariationalObjective,
    problem,
    params,
    restructure,
)
    if haskey(state_init, :objective)
        state_init.objective
    else
        init(rng, objective, problem, params, restructure)
    end
end

eachsample(samples::AbstractMatrix) = eachcol(samples)

function catsamples_and_acc(
    state_curr::Tuple{<:AbstractArray,<:Real}, state_new::Tuple{<:AbstractVector,<:Real}
)
    x = hcat(first(state_curr), first(state_new))
    ∑y = last(state_curr) + last(state_new)
    return (x, ∑y)
end
